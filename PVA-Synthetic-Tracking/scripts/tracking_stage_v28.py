"""Opt-in, source-guarded tracking-stage batching; no default/runtime edits."""
import hashlib
import heapq
import inspect
from pathlib import Path
import textwrap

import numpy as np

from tracking_batch_v27 import BatchGeometryV27, BATCH_START, BATCH_NEW, START
from tracking_geometry_v20 import OLD, REFERENCE_SHA


def predict_all(manager, timestamp_ns):
    """Same ordered arithmetic and owned covariance, fewer per-track calls."""
    from tiny_target.tracking.kalman import TemporalTrackingError

    terms_cache, covariance_cache = {}, {}
    for track in manager._tracks.values():
        delta_ns = timestamp_ns - track.state_timestamp_ns
        dt = delta_ns / 1e9
        if dt < 0:
            raise TemporalTrackingError("track prediction timestamp moved backward")
        terms = terms_cache.get(delta_ns)
        if terms is None:
            transition = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
            assert manager.config.acceleration_process_sigma_px_s2 is not None
            variance = manager.config.acceleration_process_sigma_px_s2 ** 2
            process = variance * np.array(
                [[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                 [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                 [dt ** 3 / 2, 0, dt ** 2, 0],
                 [0, dt ** 3 / 2, 0, dt ** 2]], np.float64)
            terms = transition, process
            terms_cache[delta_ns] = terms
        transition, process = terms
        mean = transition @ track.mean
        key = delta_ns, track.covariance.tobytes()
        covariance = covariance_cache.get(key)
        if covariance is None:
            covariance = transition @ track.covariance @ transition.T + process
            covariance = 0.5 * (covariance + covariance.T)
            covariance_cache[key] = covariance
        track.mean, track.covariance = mean, covariance.copy()
        track.state_timestamp_ns = timestamp_ns
        track.age_windows += 1


def innovation_batch(tracks, measurement_covariance, gaussian):
    """Update-local byte-unique matrices; None requests original scalar code."""
    if not tracks:
        return {}
    dimension = measurement_covariance.shape[0]
    covariances = [t.covariance for _, t in sorted(tracks.items())]
    if (len(tracks) > 512 or dimension not in (2, 4)
            or any(type(c) is not np.ndarray or c.dtype != np.float64
                   or c.shape != (4, 4) or not c.flags.c_contiguous for c in covariances)
            or not np.isfinite(measurement_covariance).all()):
        return None
    # Original code validates/inverts even tracks with no geometrically gated
    # pair. Never use the geometry result to omit an innovation covariance.
    keys, unique, indices = {}, [], []
    for covariance in covariances:
        key = covariance.tobytes()
        index = keys.get(key)
        if index is None:
            index = len(unique)
            keys[key] = index
            unique.append(covariance[:dimension, :dimension] + measurement_covariance)
        indices.append(index)
    stacked = np.array(unique)
    if not np.isfinite(stacked).all():
        return None
    try:
        inverses = np.linalg.inv(stacked)
    except np.linalg.LinAlgError:
        # Run the original ordered scalar path to preserve its exception and
        # any diagnostics before the first singular covariance.
        return None
    volumes = np.linalg.slogdet(stacked)[1] if gaussian else None
    return {tid: (inverses[i], float(volumes[i]) if gaussian else None)
            for tid, i in zip(sorted(tracks), indices)}


class VictimIndex:
    """One ordered heap per cell, preserving the original total victim key."""
    def __init__(self, tracks, track_cells, replaceable):
        self.cells = {}
        for tid in replaceable:
            t = tracks[tid]
            self.cells.setdefault(track_cells[tid], []).append(
                (-t.missed_windows, t.independent_confirmation_hits, tid))
        for values in self.cells.values():
            heapq.heapify(values)

    def take(self, occupancy, minimum_exclusive):
        best = None
        for cell, values in self.cells.items():
            if occupancy[cell] <= minimum_exclusive:
                continue
            missed, hits, tid = values[0]
            key = occupancy[cell], -missed, -hits, -tid
            if best is None or key > best[0]:
                best = key, cell
        if best is None:
            return None
        cell = best[1]
        tid = heapq.heappop(self.cells[cell])[2]
        if not self.cells[cell]:
            del self.cells[cell]
        return tid


def record_values(mean, covariance):
    if (type(mean) is np.ndarray and type(covariance) is np.ndarray
            and mean.dtype == covariance.dtype == np.float64
            and mean.shape == (4,) and covariance.shape == (4, 4)):
        return tuple(mean.tolist()), tuple(map(tuple, covariance.tolist()))
    return (tuple(float(v) for v in mean),
            tuple(tuple(float(v) for v in row) for row in covariance))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Changed v28 source anchor: " + old[:70])
    return source.replace(old, new)


class TrackingStageV28:
    def __init__(self, library):
        self.geometry = BatchGeometryV27(library)
        self.innovation_batches = self.innovation_fallbacks = 0
        self.innovation_tracks = 0

    def prepare(self, tracks, noise, gaussian):
        result = innovation_batch(tracks, noise, gaussian)
        if result is None:
            self.innovation_fallbacks += 1
        elif tracks:
            self.innovation_batches += 1
            self.innovation_tracks += len(tracks)
        return result

    def adapt(self, geometry_method):
        from tiny_target.tracking.kalman import KalmanTrackManager
        original = KalmanTrackManager.update
        if hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest() != REFERENCE_SHA:
            raise ValueError("Unknown frozen tracker")
        # v27 validates the exact v20 code object, not just a function name.
        self.geometry.adapt(geometry_method)
        source = inspect.getsource(original)
        source = replace_once(source, START, BATCH_START)
        source = replace_once(source, OLD, BATCH_NEW)
        source = replace_once(source, '''        prediction_cache = {}
        for track in self._tracks.values():
            self._predict(track, batch.reference_timestamp_ns, cache=prediction_cache)''',
            '''        _predict_all_v28(self, batch.reference_timestamp_ns)''')
        source = replace_once(source, '        innovation_cache = {}', '''        innovation_cache = {}
        _innovations_v28 = _prepare_v28(self._tracks, measurement_covariance,
            self.config.association_cost == "gaussian_nll")''')
        begin = source.index('            innovation_covariance = (')
        end = source.index('            inverse_innovation, log_volume = cached', begin)
        block = source[begin:end]
        source = source[:begin] + '''            if _innovations_v28 is not None:
                cached = _innovations_v28[track_id]
            else:
''' + textwrap.indent(block, '    ') + source[end:]
        source = replace_once(source, 'self._admit_births(', '_births_v28(self, ')
        source = replace_once(source, 'self._record(track,', '_record_v28(self, track,')

        birth = inspect.getsource(KalmanTrackManager._admit_births)
        birth = replace_once(birth, '        born, rejected, evictions = [], [], []',
            '''        _victims = _VictimIndex_v28(self._tracks, track_cells, replaceable)
        born, rejected, evictions = [], [], []''')
        first = birth.index('                eligible = [')
        last = birth.index('                previous = self._tracks.pop(victim)', first)
        birth = birth[:first] + '''                victim = _victims.take(occupancy, occupancy[chosen] + 1)
                if victim is None:
                    rejected.append(batch.candidates[index].candidate_index)
                    continue
''' + birth[last:]
        record = inspect.getsource(KalmanTrackManager._record)
        record = replace_once(record, '        return TrackRecord(',
            '        _mean, _covariance = _record_values_v28(track.mean, track.covariance)\n        return TrackRecord(')
        record = replace_once(record, '''            state_xy_vx_vy=tuple(float(value) for value in track.mean),
            covariance=tuple(
                tuple(float(value) for value in row) for row in track.covariance
            ),''', '''            state_xy_vx_vy=_mean,
            covariance=_covariance,''')
        scope = dict(original.__globals__, _batch_v27=self.geometry,
            _predict_all_v28=predict_all, _prepare_v28=self.prepare,
            _VictimIndex_v28=VictimIndex, _record_values_v28=record_values)
        sources = [textwrap.dedent(s) for s in (birth, record, source)]
        exec(compile(sources[0], '<tracking_stage_v28:births>', 'exec'), scope)
        exec(compile(sources[1], '<tracking_stage_v28:records>', 'exec'), scope)
        scope.update(_births_v28=scope['_admit_births'], _record_v28=scope['_record'])
        exec(compile(sources[2], '<tracking_stage_v28:update>', 'exec'), scope)
        self.transformed_sha256 = hashlib.sha256('\n'.join(sources).encode()).hexdigest()
        return scope['update']
