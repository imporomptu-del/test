"""Frozen, metadata/native-snapshot-only auxiliary replay; never decodes media.

The ordinary V28 tracker is the sole primary. Auxiliary state receives detached
priors and cannot write primary state, assignments, quality or learning inputs.
Previously archived capture coverage is conditional and is not a fresh dataset.
"""
from contextlib import ExitStack
from copy import deepcopy
import argparse
import gzip
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path('/tmp/seaqr_weak_shadow_v1_LveJSx')
ORIGINAL_FREEZE = 'e93849b1e29b8ef6a18fe23688ece8f8c959b72191081a1abb2f59af11eca8b2'
ORIGINAL_PLAN = '3156b6f47383cbdf68375728cd3eed5e5fdd13dfec4d2578f538eb2df6e91395'
CONFIG_SHA = '7c473765048e8e7f8c87042a421e0b22daf6280bb4591fd50f1d438ba2597d2f'
MOTION_SHA = 'fe450546af91f01a0fb090d76df3ba4db24081b0077c6a220d5194990fdda5b1'
SOURCES = {
    '0029': (687, '0330bc3e390a793c2bf6afe7b16720ad3cd6bb8943ee162caf9ce6eff800f359'),
    '0126': (674, 'c5302b873656793da47f1da3c03f05df595f17c3f9bc407ce0bfd99b7e718344'),
    '0055': (689, 'c59abb3dad5c8787aab8a5a86eff466928a918be79a536960a7713d8a5dc539f'),
}
WINDOWS = {'0029': [[70,81],[270,281],[344,355]],
    '0126': [[16,28],[200,218],[258,270],[289,301],[324,336],[449,461],[478,490],[522,534]],
    '0055': [[100,111],[300,311],[500,511]]}
PINS = {
    'replay_weak_divergence_v1.py': '8a8ff3fa357be07999cb9a93c6cf3099be968534f903d49482638b7e5c35be82',
    'combined_v29_state.py': '728bf8f6b0b958b6cd56eeda62425b921d1abb7130a9832ce5048124ab4471d5',
    'replay_tracking_v27.py': 'fbaefd79d78bdd4dad808fe6acf47338a1500a38c025d35141e2a16c407e35c5',
    'accuracy_v56_capture.py': 'b9f38440987fcf85ae39c77e8a6244eaed0d742d9b978f90476901bb0caffae0',
    'weak_continuation_information_v1.py': '7fab347bc47d51df96c73cae3efb6a89266cde77938d3c0b5ddbb23a8f26ff15',
    'weak_continuation_shadow_v1.py': '4d15015eee7a15adf177d25cd947c5f59b829a01c6a2cb960d83a3aa721e8d09',
    'profile_visible_v17.py': '233e5cb5010be48a2aa45cb9c22e395155903af3df690840809d6cca3a91647f',
}
REQUIRED = set(PINS) | {'run_weak_auxiliary_replay_v1.py', 'test_run_weak_auxiliary_replay_v1.py',
    'weak_continuation_auxiliary_v1.py', 'test_weak_continuation_auxiliary_v1.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def encoded(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        require(math.isfinite(result), 'Nonfinite JSON number')
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=number,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


class BoundFiles:
    """Symlink-free, caller-bound files; remember every read and rehash later."""
    def __init__(self, root):
        self.root = Path(root)
        require(self.root.is_absolute() and self.root.resolve() == self.root and
                self.root.is_dir(), 'Canonical input directory required')
        self.hashes = {}

    def path(self, name, expected):
        part = Path(name)
        require(type(name) is str and name and not part.is_absolute() and '..' not in part.parts
                and str(part) == name and digest_ok(expected), 'Unsafe or unbound input')
        target = self.root
        for p in part.parts:
            target /= p
            require(not target.is_symlink(), 'Symlink input refused')
        require(target.is_file() and sha(target) == expected, 'Bound input changed: '+name)
        require(name not in self.hashes or self.hashes[name] == expected, 'Conflicting input binding')
        self.hashes[name] = expected
        return target

    def read(self, name, expected):
        path = self.path(name, expected)
        value = loads(path.read_text())
        self.path(name, expected)
        require(type(value) is dict, 'JSON object required')
        return value

    def unchanged(self):
        for name, digest in list(self.hashes.items()):
            self.path(name, digest)


def load_freeze(path, expected_sha):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.name == 'freeze.json', 'Canonical freeze.json required')
    bound = BoundFiles(path.parent)
    frozen = bound.read(path.name, expected_sha)
    require(frozen.get('schema') == 'seaqr.weak-auxiliary-replay.freeze.v1' and frozen.get('pre_run') is True,
            'Pre-run auxiliary freeze required')
    require(frozen.get('original_root') == str(ORIGINAL_ROOT)
            and frozen.get('original_freeze_sha256') == ORIGINAL_FREEZE
            and frozen.get('original_plan_sha256') == ORIGINAL_PLAN, 'Original experiment differs')
    audits = frozen.get('original_audits_sha256', {})
    require(set(audits) == set(SOURCES) and all(digest_ok(h) for h in audits.values()), 'All original audit pins required')
    files = frozen.get('files_sha256', {})
    require(type(files) is dict and REQUIRED <= set(files), 'Incomplete frozen source bundle')
    for name, digest in files.items():
        require(Path(name).name == name, 'Flat literal bundle filenames required')
        bound.path(name, digest)
    require(all(files[name] == digest for name, digest in PINS.items()), 'Frozen helper source changed')
    require(sha(__file__) == files['run_weak_auxiliary_replay_v1.py'], 'Running source is not frozen source')
    for key in ('geometry', 'batch_library'):
        item = frozen.get(key, {})
        require(set(item) == {'path','sha256'} and digest_ok(item['sha256']), 'Native library binding required')
        p = Path(item['path'])
        require(p.is_absolute() and p.resolve() == p and p.is_file() and sha(p) == item['sha256'], 'Native library changed')
    return frozen, bound


def original_inputs(directory, clip, audit_sha256, frozen):
    directory = Path(directory)
    require(directory == ORIGINAL_ROOT and clip in SOURCES, 'Only original three exposed clips allowed')
    require(audit_sha256 == frozen['original_audits_sha256'][clip], 'Caller audit differs from new freeze')
    root = BoundFiles(directory)
    old = root.read('freeze.json', frozen['original_freeze_sha256'])
    plan = root.read('plan.json', frozen['original_plan_sha256'])
    require(old.get('schema') == 'seaqr.weak-continuation-shadow.freeze.v1' and old.get('pre_run') is True
            and old.get('plan_sha256') == ORIGINAL_PLAN, 'Original freeze invalid')
    require(plan.get('schema') == 'seaqr.weak-continuation-shadow.plan.v1' and set(plan.get('clips', {})) == set(SOURCES), 'Original scope differs')
    for cid, (count, source) in SOURCES.items():
        spec = plan['clips'][cid]
        require(spec['frames'] == count and spec['source_sha256'] == source
                and spec['weak_windows_inclusive'] == WINDOWS[cid], 'Original cohort or schedule differs')
    audit = root.read(clip+'/independent_audit.json', audit_sha256)
    count, source = SOURCES[clip]
    for key, value in dict(schema='seaqr.weak-continuation-shadow.audit.v1', passed=True, clip=clip,
                          frames=count, source_sha256=source, freeze_sha256=ORIGINAL_FREEZE,
                          plan_sha256=ORIGINAL_PLAN, baseline_journal_non_timing_exact=True,
                          baseline_output_state_learning_digests_exact=True, native_state_guards_unchanged=True,
                          production_changed=False, weak_learning_enabled=False).items():
        require(audit.get(key) == value and type(audit.get(key)) is type(value), 'Original audit guard differs: '+key)
    evidence = BoundFiles(directory/clip)
    pins = audit.get('files_sha256', {})
    require(type(pins) is dict and all(digest_ok(h) for h in pins.values()), 'Missing audited artifact map')
    for name, digest in pins.items():
        if not name.startswith('shadow/captures/'):
            evidence.path(name, digest)
    reference = evidence.read('clean/launch.json', pins['clean/launch.json'])
    require(reference['source_sha256'] == source and reference['config_sha256'] == CONFIG_SHA
            and reference['motion_config_sha256'] == MOTION_SHA and reference['expected_frames'] == count
            and reference['max_frames'] is None and reference['fps'] == 10
            and reference['annotations_supplied_to_detector'] is False, 'Original full baseline launch differs')
    receipt = evidence.read('clean.shadow.json', pins['clean.shadow.json'])
    require(receipt['schema'] == 'seaqr.weak-continuation-shadow.run.v1' and receipt['passed'] is True
            and receipt['error'] is None and receipt['arm'] == 'clean' and receipt['clip'] == clip
            and receipt['processed_frames'] == receipt['expected_frames'] == count
            and receipt['freeze_sha256'] == ORIGINAL_FREEZE and receipt['plan_sha256'] == ORIGINAL_PLAN,
            'Original clean receipt differs')
    require(len(receipt['baseline_digests']) == count and all(row['frame'] == i for i,row in enumerate(receipt['baseline_digests'])),
            'Incomplete original private-state digests')
    return root, evidence, audit, reference, receipt


def load_helpers(frozen, bundle):
    """Called only after source/native hash verification; no media dependencies."""
    sys.path.insert(0, str(bundle.root))
    names = ('replay_weak_divergence_v1', 'replay_tracking_v27', 'combined_v29_state',
             'accuracy_v56_capture', 'weak_continuation_auxiliary_v1')
    result = {}
    for name in names:
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == bundle.root/(name+'.py')
                and sha(module.__file__) == frozen['files_sha256'][name+'.py'], 'Imported nonfrozen helper '+name)
        result[name] = module
    return result


def primary_forecasts(tracker, frame, timestamp, segment):
    """Pure detached strong-only predictions; no auxiliary budget or old IDs."""
    import numpy as np
    result = []
    for polarity, manager in tracker.managers.items():
        if manager._segment_index != segment or manager._last_timestamp_ns is None:
            continue
        if (timestamp-manager._last_timestamp_ns)/1e9 > manager.config.maximum_timestamp_gap_s:
            continue
        for tid, track in sorted(manager._tracks.items()):
            mean, covariance = manager._predicted_state(track, timestamp)
            noise = manager._measurement_covariance()
            innovation = covariance[:2,:2]+noise
            require(mean.shape == (4,) and covariance.shape == (4,4) and noise.shape == (2,2)
                    and np.isfinite(mean).all() and np.isfinite(covariance).all()
                    and np.array_equal(covariance, covariance.T), 'Invalid detached primary forecast')
            np.linalg.cholesky(covariance); np.linalg.cholesky(innovation)
            qualified = f'{polarity}:{tid}' in tracker.qualified
            age = (timestamp-track.last_measurement_timestamp_ns)/1e9
            result.append(dict(identity=f'{segment}/{polarity}:{tid}', polarity=polarity, track_id=tid,
                segment=segment, frame_index=frame, timestamp_ns=timestamp,
                reference_xy=mean[:2].tolist(), innovation_covariance_2x2=innovation.tolist(),
                predicted_mean=mean.tolist(), predicted_covariance=covariance.tolist(),
                strong_measurement_covariance=noise.tolist(), prior_qualified=qualified,
                prior_last_strong_timestamp_ns=track.last_measurement_timestamp_ns,
                prior_associated_update_count=track.associated_update_count,
                prior_independent_confirmation_hits=track.independent_confirmation_hits,
                prior_confirmation_timestamp_ns=track.confirmation_timestamp_ns,
                weak_budget_available=True, strong_age_seconds=age,
                query_eligible=qualified and track.confirmation_timestamp_ns is not None and 0 < age <= tracker.config.coast_seconds))
    return result


def geometrically_covers(rectangle, forecast, shape):
    """Conservative full 45px disk and 5x5 neighborhood; no pixel peeking."""
    require(rectangle['shape_hw'] == list(shape), 'Capture native shape differs')
    x,y = forecast['reference_xy']; h,w = shape
    tile = rectangle['tile_bounds_exclusive_xyxy']; capture = rectangle['capture_bounds_exclusive_xyxy']
    if not (x-45 >= tile[0] and y-45 >= tile[1] and x+45 <= tile[2]-1 and y+45 <= tile[3]-1):
        return False
    return (capture[0] <= max(0,math.ceil(x-45)-2) and capture[1] <= max(0,math.ceil(y-45)-2)
            and capture[2] > min(w-1,math.floor(x+45)+2) and capture[3] > min(h-1,math.floor(y+45)+2))


class CaptureProvider:
    """First lexicographic geometrically complete capture, never outcome search."""
    def __init__(self, evidence, pins, row, shape, enabled, validate_rectangle):
        self.evidence, self.pins = evidence, pins
        self.frame, self.segment, self.shape = row['frame_index'], row['segment'], list(shape)
        self.enabled, self.calls, self.inventory = enabled, [], {}
        for desc in row['capture_files']:
            # Deliberately discard archived forecast identities.
            clean = {key:desc[key] for key in ('path','sha256','metadata_path','metadata_sha256')}
            for key, digest_key in (('path','sha256'),('metadata_path','metadata_sha256')):
                name = clean[key]
                require(name.startswith('captures/') and pins.get('shadow/'+name) == clean[digest_key], 'Capture missing original audit binding')
            key = clean['path']
            require(key not in self.inventory or self.inventory[key][0] == clean, 'Conflicting reused capture descriptor')
            if key not in self.inventory:
                meta = evidence.read('shadow/'+clean['metadata_path'], clean['metadata_sha256'])
                require(meta.get('frame') == self.frame and meta.get('segment') == self.segment
                        and meta.get('prelearning') is True and meta.get('baseline_learning_unchanged') is True
                        and meta.get('read_only_capture') is True and meta.get('production_selection_unchanged') is True,
                        'Capture provenance differs')
                validate_rectangle(meta['rectangle'])
                self.inventory[key] = clean, meta
        self.cache = {}

    def __call__(self, forecast):
        import numpy as np
        require(forecast['frame_index'] == self.frame and forecast['segment'] == self.segment, 'Wrong current capture query')
        row = dict(query_identity=forecast['identity'], forecast=deepcopy(forecast), status=None, descriptor=None)
        self.calls.append(row)
        if not self.enabled:
            row['status'] = 'outside_frozen_window'; return None
        selected = next(((desc,meta) for _,(desc,meta) in sorted(self.inventory.items())
                         if geometrically_covers(meta['rectangle'], forecast, self.shape)), None)
        if selected is None:
            row['status'] = 'no_geometrically_complete_cached_capture'; return None
        desc,meta = selected
        row.update(status='geometrically_complete_capture_supplied', descriptor=deepcopy(desc))
        key = desc['path']
        if key not in self.cache:
            path = self.evidence.path('shadow/'+key, desc['sha256'])
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                require(len({e.filename for e in entries}) == len(entries)
                        and sum(e.file_size for e in entries) <= 128*1024*1024
                        and {'values.npy','flags.npy'} <= {e.filename for e in entries}, 'Unbounded/invalid capture archive')
            with np.load(path, allow_pickle=False) as data:
                values, flags = data['values'].copy(), data['flags'].copy()
            b = meta['rectangle']['capture_bounds_exclusive_xyxy']; shape = (b[3]-b[1],b[2]-b[0])
            require(values.dtype == np.float32 and flags.dtype == np.uint8
                    and values.shape == shape+(len(meta['float_fields']),)
                    and flags.shape == shape+(len(meta['flag_fields']),), 'Capture array shape/dtype differs')
            values.flags.writeable = False; flags.flags.writeable = False
            self.cache[key] = values,flags
            self.evidence.path('shadow/'+key, desc['sha256'])
        values,flags = self.cache[key]
        # Copies prevent the consumer from changing a later query's snapshot.
        values,flags = values.copy(),flags.copy()
        values.flags.writeable = False; flags.flags.writeable = False
        return dict(values=values, flags=flags, metadata=deepcopy(meta))


def snapshot_primary(tracker, state_module, digest_module):
    normalized = digest_module.normalized(state_module.state_of(tracker))
    return normalized, hashlib.sha256(encoded(normalized).encode()).hexdigest()


def replay_frame(tracker, auxiliary, baseline, trace, reference_digest, provider, modules):
    """Generated-test seam; caller owns pinned native update context and inputs."""
    import numpy as np
    h = modules['replay_weak_divergence_v1']; d = modules['replay_tracking_v27']; st = modules['combined_v29_state']
    frame,stamp,segment = (baseline[k] for k in ('frame_index','timestamp_ns','segment'))
    require(trace['frame_index'] == frame and trace['timestamp_ns'] == stamp and trace['segment'] == segment, 'Trace coordinate sequence differs')
    proposals = deepcopy(baseline['candidates'])
    h.compare([{k:v for k,v in p.items() if k != 'source_xy'} for p in proposals],
              [{k:v for k,v in p.items() if k != 'source_xy'} for p in trace['strong_proposals']], 'strong proposal stream')
    matrix=np.asarray(baseline['source_to_reference'],np.float64); shape=tuple(baseline['coverage']['full_shape_hw'])
    require(matrix.shape == (3,3) and np.isfinite(matrix).all(), 'Invalid current geometry')
    before_prepare, before_prepare_digest = snapshot_primary(tracker,st,d)
    learning = deepcopy(tracker.learning_centers(stamp,segment))
    require(d.digest(learning) == reference_digest['learning'], 'Original preupdate learning digest differs')
    priors=primary_forecasts(tracker,frame,stamp,segment)
    aux_before=auxiliary.snapshot()
    queries=auxiliary.prepare(frame,stamp,segment,deepcopy(priors))
    after_prepare, after_prepare_digest = snapshot_primary(tracker,st,d)
    require(before_prepare == after_prepare and before_prepare_digest == after_prepare_digest, 'Auxiliary prepare mutated primary')
    records,metrics=tracker.update(deepcopy(proposals),frame,stamp,segment,matrix.copy(),shape)
    h.compare_records(records,baseline['tracks'],f'frame{frame}.primary_records')
    h.compare(metrics,baseline['tracking_metrics'],f'frame{frame}.primary_metrics',True)
    state_before,state_digest_before=snapshot_primary(tracker,st,d)
    output_normalized_before=d.normalized([records,metrics])
    output_before=d.digest([records,metrics])
    require(reference_digest == dict(frame=frame,output=output_before,state=state_digest_before,learning=d.digest(learning)),
            'Original output/private-state/learning digest differs')
    next_learning=deepcopy(tracker.learning_centers(stamp+100000000,segment))
    post=[dict(identity=f'{segment}/{polarity}:{r["track_id"]}',polarity=polarity,**r)
          for polarity,manager in tracker.managers.items() for r in h.manager_state(manager,stamp)]
    aux_records,aux_metrics=auxiliary.step(deepcopy(records),deepcopy(proposals),matrix.copy(),shape,provider)
    state_after,state_digest_after=snapshot_primary(tracker,st,d)
    output_normalized_after=d.normalized([records,metrics])
    output_after=d.digest([records,metrics]); next_learning_after=tracker.learning_centers(stamp+100000000,segment)
    require(state_before == state_after and state_digest_before == state_digest_after
            and output_normalized_before == output_normalized_after and output_before == output_after and next_learning == next_learning_after
            and d.digest(next_learning) == d.digest(next_learning_after), 'Auxiliary step/provider changed primary or learning')
    primary=dict(frame_index=frame,timestamp_ns=stamp,segment=segment,records=records,metrics=metrics,
        learning_centers=learning,next_learning_centers_before_aux=next_learning,next_learning_centers_after_aux=next_learning_after,
        primary_output_normalized_before_aux=output_normalized_before,primary_output_normalized_after_aux=output_normalized_after,
        actual_learning_normalized=d.normalized(learning),
        next_learning_normalized_before_aux=d.normalized(next_learning),next_learning_normalized_after_aux=d.normalized(next_learning_after),
        original_digest_reference=reference_digest,primary_prepare_state_sha256_before=before_prepare_digest,
        primary_prepare_state_sha256_after=after_prepare_digest,primary_state_before_aux=state_before,
        primary_state_after_aux=state_after,primary_state_sha256_before_aux=state_digest_before,
        primary_state_sha256_after_aux=state_digest_after,primary_output_sha256_before_aux=output_before,
        primary_output_sha256_after_aux=output_after,learning_sha256_before_aux=d.digest(next_learning),
        learning_sha256_after_aux=d.digest(next_learning_after),actual_learning_sha256=d.digest(learning))
    diagnostic=dict(frame_index=frame,timestamp_ns=stamp,segment=segment,primary_forecasts=priors,
        auxiliary_queries=queries,primary_post_strong=post,primary_records_after_strong=deepcopy(records),
        auxiliary_records=aux_records,auxiliary_metrics=aux_metrics,capture_calls=deepcopy(provider.calls),
        aux_state_before=aux_before,aux_state_after=auxiliary.snapshot(),capture_scheduled=provider.enabled)
    return primary,diagnostic


def run(directory,clip,audit_sha256,freeze,freeze_sha256,output):
    frozen,bundle=load_freeze(freeze,freeze_sha256)
    original,evidence,audit,reference,clean_receipt=original_inputs(directory,clip,audit_sha256,frozen)
    output=Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and not output.is_symlink() and ORIGINAL_ROOT not in output.parents
            and output != bundle.root and output not in bundle.root.parents, 'Fresh isolated absolute output required')
    modules=load_helpers(frozen,bundle)
    h=modules['replay_weak_divergence_v1']
    runtime=h.verify_runtime(reference)
    from tiny_target.visible_baseline import VisibleConfig,VisibleTracks
    from tiny_target.tracking.kalman import KalmanTrackManager
    config=VisibleConfig(**reference['configuration'])
    primary=VisibleTracks(config,reference['fps'])
    auxiliary=modules['weak_continuation_auxiliary_v1'].AuxiliaryGapSupport(config,reference['fps'])
    output.mkdir(parents=False)
    receipt=dict(schema='seaqr.weak-auxiliary-replay.run.v1',passed=False,error=None,clip=clip,
        expected_frames=SOURCES[clip][0],processed_frames=0,source_sha256=SOURCES[clip][1],
        original_directory=str(Path(directory)/clip),original_audit_sha256=audit_sha256,
        freeze_sha256=freeze_sha256,original_freeze_sha256=ORIGINAL_FREEZE,original_plan_sha256=ORIGINAL_PLAN,
        runtime_hashes=runtime,production_changed=False,weak_learning_enabled=False,media_accessed=False,
        detector_rerun=False,old_shadow_forecasts_or_accepted_weak_events_used=False,
        coverage_policy='First lexicographic complete45px geometry capture; unknown coverage abstains; no stitching or alternate after pixel inspection',
        limitations=['Exposed development clips, not holdout validation.',
            'Cached rectangles were selected by a previous shadow; availability is conditional and missing coverage is not negative evidence.'])
    raw_hashes={name:hashlib.sha256() for name in ('clean','primary','auxiliary')}
    raw_bytes={name:0 for name in raw_hashes}
    try:
        with ExitStack() as stack:
            method,optimized,scalar=stack.enter_context(h.native_backend(frozen['geometry']['path'],frozen['geometry']['sha256'],
                frozen['batch_library']['path'],frozen['batch_library']['sha256']))
            stack.enter_context(patch.object(KalmanTrackManager,'update',method))
            clean=stack.enter_context(evidence.path('clean/frames.jsonl',audit['files_sha256']['clean/frames.jsonl']).open('rb'))
            trace=stack.enter_context(evidence.path('shadow/shadow_trace.jsonl',audit['files_sha256']['shadow/shadow_trace.jsonl']).open('rb'))
            outputs={name:stack.enter_context(gzip.open(output/(name+'.jsonl.gz'),'xb',compresslevel=1)) for name in ('clean','primary','auxiliary')}
            for frame,lines in enumerate(itertools.zip_longest(clean,trace)):
                require(frame<SOURCES[clip][0] and all(line and line.strip() for line in lines), 'Unequal, blank or excessive full-clip journals')
                b,t=map(loads,lines)
                require(b['frame_index']==frame and b['timestamp_ns']==frame*100000000, 'Noncontiguous full replay')
                enabled=any(lo<=frame<=hi for lo,hi in WINDOWS[clip])
                require(t['capture_scheduled'] is enabled, 'Old capture schedule differs')
                provider=CaptureProvider(evidence,audit['files_sha256'],t,b['coverage']['full_shape_hw'],enabled,
                    modules['accuracy_v56_capture'].validate_rectangle)
                p,a=replay_frame(primary,auxiliary,b,t,clean_receipt['baseline_digests'][frame],provider,modules)
                encoded_rows=dict(clean=lines[0],primary=(encoded(p)+'\n').encode(),auxiliary=(encoded(a)+'\n').encode())
                for name,raw in encoded_rows.items():
                    outputs[name].write(raw);raw_hashes[name].update(raw);raw_bytes[name]+=len(raw)
                receipt['processed_frames']=frame+1
                require(not any(note.get('status') == 'invalid_capture_or_weak_covariance'
                                for note in a['auxiliary_metrics'].get('decisions', [])),
                        'Invalid capture/weak covariance diagnostic; preserved row, abort without retry')
                if frame%100==0: print(clip,frame+1,'/',SOURCES[clip][0],flush=True)
            require(receipt['processed_frames']==SOURCES[clip][0], 'Incomplete full replay')
            require(raw_hashes['clean'].hexdigest()==audit['files_sha256']['clean/frames.jsonl'], 'Raw clean archive copy changed')
        receipt['native_counters']=dict(geometry_calls=scalar.calls,scalar_fallbacks=scalar.fallbacks,
            optimized_fallbacks=optimized.geometry.fallbacks,innovation_fallbacks=optimized.innovation_fallbacks)
        evidence.unchanged();original.unchanged();bundle.unchanged()
        require(h.verify_runtime(reference)==runtime, 'Runtime changed')
        receipt['passed']=True
    except BaseException as error:
        receipt['error']=dict(type=type(error).__name__,message=str(error),detail=getattr(error,'detail',None))
        raise
    finally:
        receipt['input_files_sha256']={str(evidence.root/name):digest for name,digest in evidence.hashes.items()}
        receipt['original_root_files_sha256']={str(original.root/name):digest for name,digest in original.hashes.items()}
        receipt['bundle_files_sha256']={str(bundle.root/name):digest for name,digest in bundle.hashes.items()}
        receipt['artifacts_sha256']={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        receipt['artifact_uncompressed']={name+'.jsonl.gz':dict(sha256=raw_hashes[name].hexdigest(),
            bytes=raw_bytes[name],frames=receipt['processed_frames']) for name in raw_hashes}
        with (output/'receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','freeze','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--clip',choices=tuple(SOURCES),required=True)
    for name in ('audit-sha256','freeze-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    run(args.directory,args.clip,args.audit_sha256,args.freeze,args.freeze_sha256,args.output)
