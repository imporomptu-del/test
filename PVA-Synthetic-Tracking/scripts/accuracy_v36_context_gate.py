"""Causal shadow-output appearance gate; never modifies detector feedback.

Evaluates only baseline-qualified actual measurements. Coasts inherit their last
measurement's result and never create image evidence. Unknown/truncated evidence
keeps baseline eligibility, explicitly flagged, rather than declaring absence.
"""
import copy
import math

import numpy as np

from accuracy_v36_context import PointEdgeDiagnostic


class CausalPointContextGate:
    def __init__(self, diagnostic=None):
        self.diagnostic = diagnostic if diagnostic is not None else PointEdgeDiagnostic()
        self._frame = -1
        self._timestamp = None
        self._segment = None
        self._shape = None
        self._state = {}

    @staticmethod
    def _integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def _validate(self, row, gray):
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or min(gray.shape) < 1:
            raise ValueError("Nonempty native uint8 grayscale frame required")
        if self._shape is not None and gray.shape != self._shape:
            raise ValueError("Changed source geometry")
        frame, timestamp, segment = row.get("frame_index"), row.get("timestamp_ns"), row.get("segment")
        if not self._integer(frame) or frame != self._frame+1:
            raise ValueError("Contiguous frames beginning at zero required")
        if not self._integer(timestamp) or (self._timestamp is not None and timestamp <= self._timestamp):
            raise ValueError("Strictly increasing nonnegative timestamps required")
        if not self._integer(segment) or not isinstance(row.get("tracks"), list):
            raise ValueError("Valid segment and track list required")
        identities = set()
        for track in row["tracks"]:
            identity = track.get("track_id")
            if (not isinstance(identity,str) or identity.split(":")[0] not in ("bright","dark")
                    or not self._integer(track.get("segment")) or track["segment"] != segment
                    or (segment,identity) in identities):
                raise ValueError("Unique polarity-bearing track IDs in current segment required")
            identities.add((segment,identity))
            if type(track.get("measured")) is not bool or type(track.get("qualified_moving")) is not bool:
                raise ValueError("Boolean track evidence and qualification required")
            xy = track.get("measurement_source_xy")
            if track["measured"]:
                if (not isinstance(xy,(list,tuple)) or len(xy)!=2
                        or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in xy)):
                    raise ValueError("Finite measured source coordinate required")
            elif xy is not None:
                raise ValueError("Prediction cannot contain a current measurement")
        return identities

    def update(self, row, gray):
        identities = self._validate(row, gray)
        frame, segment = row["frame_index"], row["segment"]
        state = ({k:v for k,v in self._state.items() if k in identities}
                 if segment == self._segment else {})
        output = {}
        h,w = gray.shape
        for track in row["tracks"]:
            key = (segment,track["track_id"])
            qualified, measured = track["qualified_moving"], track["measured"]
            if not qualified:
                if measured:
                    state.pop(key,None)
                output[key] = dict(accepted=False,reason="baseline_unqualified",measurement_frame=None,features=None)
                continue
            if not measured:
                prior = state.get(key)
                if prior is None:
                    output[key] = dict(accepted=True,reason="unknown_missing_history",measurement_frame=None,features=None)
                else:
                    accepted,reason,observed_frame = prior
                    output[key] = dict(accepted=accepted,reason="coast_"+reason,measurement_frame=observed_frame,features=None)
                continue
            sx,sy = track["measurement_source_xy"]
            x,y = math.floor(sx+.5),math.floor(sy+.5)
            features = None
            if x-12<0 or y-12<0 or x+12>=w or y+12>=h:
                accepted,reason = True,"unknown_truncated_patch"
            else:
                # Own the patch so diagnostic implementations cannot alter input.
                patch = gray[y-12:y+13,x-12:x+13].copy()
                features = self.diagnostic.measure(patch,track["track_id"].split(":")[0])
                if (not isinstance(features,dict) or type(features.get("informative")) is not bool
                        or isinstance(features.get("point_minus_edge_fraction"),bool)
                        or not isinstance(features.get("point_minus_edge_fraction"),(float,int))
                        or not math.isfinite(features["point_minus_edge_fraction"])):
                    raise ValueError("Malformed point-context diagnostic")
                if not features["informative"]:
                    accepted,reason = True,"unknown_uninformative_patch"
                elif features["point_minus_edge_fraction"] > 0:
                    accepted,reason = True,"point_preferred"
                else:
                    accepted,reason = False,"edge_preferred_or_tie"
            state[key] = (accepted,reason,frame)
            output[key] = dict(accepted=accepted,reason=reason,measurement_frame=frame,features=copy.deepcopy(features))
        # Commit only after validation and every requested fit succeeded.
        self._state = state
        self._frame = frame
        self._timestamp = row["timestamp_ns"]
        self._segment = segment
        self._shape = gray.shape
        return output

    def retained_state_counts(self):
        return dict(tracks=len(self._state))
