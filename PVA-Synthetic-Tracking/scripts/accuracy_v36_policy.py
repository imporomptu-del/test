"""Causal shadow qualification ablations, never detection/tracker inputs.

This is an experiment, not an airborne classifier. It cannot improve the frozen
detector's measurements or its association. Missing shape is unknown support,
not proof of clutter. Label files, clip IDs and review regions are not inputs.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

ARMS=('baseline','recent_support','recent_excursion','bounded_shape','combined')


@dataclass(frozen=True)
class PolicyConfig:
    window_hits: int=8
    recent_frame_window: int=8
    minimum_recent_hits: int=5
    minimum_excursion_px: float=12.0

    def __post_init__(self):
        for name in ('window_hits','recent_frame_window','minimum_recent_hits'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError('Positive integer window/count required')
        if not 2<=self.minimum_recent_hits<=min(self.window_hits,self.recent_frame_window):
            raise ValueError('Minimum hits outside both windows')
        if (type(self.minimum_excursion_px) not in (int,float)
                or not math.isfinite(self.minimum_excursion_px) or self.minimum_excursion_px<=0):
            raise ValueError('Positive finite movement threshold required')


def finite_xy(value):
    if (not isinstance(value,(list,tuple)) or len(value)!=2
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in value)):
        raise ValueError('Finite two-dimensional observation required')
    return value


def bounded_shape(value):
    if value is None or value==[]:
        return False
    if not isinstance(value,(list,tuple)):
        raise ValueError('Shape support must be a finite Nx2 sequence')
    for xy in value:finite_xy(xy)
    return bool(value)


class CausalQualification:
    def __init__(self,config=PolicyConfig()):
        if not isinstance(config,PolicyConfig):
            raise ValueError('Explicit validated policy config required')
        self.config=config
        self._states={}
        self._frame=None
        self._timestamp=None
        self._segment=None

    def retained_state_counts(self):
        return dict(tracks=len(self._states),measurements=sum(len(x['history']) for x in self._states.values()))

    def update(self,row):
        frame,timestamp,segment=(row[k] for k in ('frame_index','timestamp_ns','segment'))
        if (type(frame) is not int or frame!=(0 if self._frame is None else self._frame+1)
                or type(timestamp) is not int or timestamp<0
                or self._timestamp is not None and timestamp<=self._timestamp
                or type(segment) is not int or segment<0):
            raise ValueError('Contiguous frames and strictly increasing time required')
        matrix=np.asarray(row['source_to_reference'],dtype=float)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
            raise ValueError('Finite invertible source-to-reference mapping required')
        observations={}
        for track in row['tracks']:
            tid=track['track_id']
            if not isinstance(tid,str) or not tid or type(track['segment']) is not int or track['segment']!=segment:
                raise ValueError('Invalid track identity or segment')
            key=(segment,tid)
            if key in observations:
                raise ValueError('Duplicate track identity')
            if type(track['measured']) is not bool or type(track['qualified_moving']) is not bool:
                raise ValueError('Explicit measured/qualified booleans required')
            point=None
            shape=bounded_shape(track.get('learning_shape_reference_xy'))
            if track['measured']:
                x,y=finite_xy(track['measurement_source_xy'])
                raw=matrix@np.array([x,y,1.0],dtype=float)
                if not np.isfinite(raw).all() or abs(raw[2])<1e-12:
                    raise ValueError('Invalid homogeneous measurement mapping')
                point=(float(raw[0]/raw[2]),float(raw[1]/raw[2]))
                finite_xy(point)
            elif track.get('measurement_source_xy') is not None:
                raise ValueError('Predicted state cannot contain a measured coordinate')
            observations[key]=(track['measured'],track['qualified_moving'],point,shape)
        # No mutation before all input observations have passed validation.
        if segment!=self._segment:self._states={}
        self._states={k:v for k,v in self._states.items() if k in observations}
        outputs={};cfg=self.config
        for key,(measured,qualified,point,shape) in observations.items():
            state=self._states.setdefault(key,dict(history=deque(maxlen=max(cfg.window_hits,cfg.recent_frame_window)),shape=False))
            history=state['history']
            if measured:
                history.append((frame,*point));state['shape']=shape
            recent=sum(h[0]>=frame-cfg.recent_frame_window+1 for h in history)
            movement=list(history)[-cfg.window_hits:]
            excursion=(math.hypot(max(h[1] for h in movement)-min(h[1] for h in movement),
                max(h[2] for h in movement)-min(h[2] for h in movement)) if movement else 0.0)
            support=recent>=cfg.minimum_recent_hits
            moving=len(movement)>=cfg.minimum_recent_hits and excursion>=cfg.minimum_excursion_px
            decisions=dict(baseline=qualified,recent_support=qualified and support,
                recent_excursion=qualified and moving,bounded_shape=qualified and state['shape'],
                combined=qualified and support and moving and state['shape'])
            outputs[key]=dict(decisions=decisions,recent_hits=recent,recent_excursion_px=excursion,
                last_measurement_age_frames=frame-history[-1][0] if history else None,
                bounded_shape=state['shape'])
        self._frame,self._timestamp,self._segment=frame,timestamp,segment
        return outputs
