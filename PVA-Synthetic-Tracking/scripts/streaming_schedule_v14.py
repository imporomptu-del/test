"""Experimental causal work scheduling only; not a detector or a truth consumer."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScheduleConfig:
    height: int = 3190
    width: int = 4784
    tile: int = 256
    halo: int = 8
    cap: int = 32
    blind: int = 8
    seed_radius: int = 45
    ttl: int = 16
    window: int = 16
    stride: int = 8
    warmup: int = 4

    def __post_init__(self):
        for name in ('height', 'width', 'tile', 'cap', 'blind', 'ttl', 'window', 'stride'):
            v = getattr(self, name)
            if type(v) is not int or v <= 0:
                raise ValueError('Positive integer required: '+name)
        for name in ('halo', 'seed_radius', 'warmup'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Nonnegative integer required: '+name)
        if not self.blind <= self.cap <= self.tiles or self.stride > self.window or self.window != 16:
            raise ValueError('Invalid resource bounds')

    @property
    def nx(self):
        return math.ceil(self.width/self.tile)

    @property
    def tiles(self):
        return self.nx*math.ceil(self.height/self.tile)

    def rect(self, index, halo=False):
        if type(index) is not int or not 0 <= index < self.tiles:
            raise ValueError('Invalid tile')
        x = index % self.nx*self.tile
        y = index // self.nx*self.tile
        pad = self.halo if halo else 0
        return (max(0, x-pad), max(0, y-pad),
                min(self.width, x+self.tile+pad), min(self.height, y+self.tile+pad))

    def requested(self, x, y):
        if not all(math.isfinite(v) for v in (x, y)) or not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError('Seed outside finite image coordinates')
        r = self.seed_radius
        x0, x1 = max(0, math.floor(x-r)), min(self.width-1, math.floor(x+r))
        y0, y1 = max(0, math.floor(y-r)), min(self.height-1, math.floor(y+r))
        return [iy*self.nx+ix for iy in range(y0//self.tile, y1//self.tile+1)
                for ix in range(x0//self.tile, x1//self.tile+1)]


def required_halo(timestamps_ns, maximum_speed_component=3.):
    if (len(timestamps_ns) != 16 or any(type(t) is not int for t in timestamps_ns)
            or any(b <= a for a, b in zip(timestamps_ns, timestamps_ns[1:]))
            or not math.isfinite(maximum_speed_component) or maximum_speed_component <= 0):
        raise ValueError('Strictly increasing 16-frame timestamps and finite speed required')
    midpoint = (timestamps_ns[0]+timestamps_ns[-1])//2
    return math.ceil(max(midpoint-timestamps_ns[0], timestamps_ns[-1]-midpoint)
                     /1e9*maximum_speed_component)+1  # bilinear neighbor


class Scheduler:
    def __init__(self, config=ScheduleConfig()):
        self.cfg = config
        self.segment = None
        self.last_index = None
        self.last_timestamp = None
        self.history = []
        self.count = 0
        self.seeds = {}
        self.visits = [-1]*config.tiles

    def update(self, index, timestamp_ns, segment, seeds):
        # Validate everything BEFORE committing state. No implicit gap recovery.
        if (any(type(v) is not int or v < 0 for v in (index, timestamp_ns, segment))
                or (self.last_index is not None and index != self.last_index+1)
                or (self.last_timestamp is not None and timestamp_ns <= self.last_timestamp)):
            raise ValueError('Contiguous frames and strictly increasing timestamps required')
        incoming = []
        for seed in seeds:
            score = seed['score']
            if not math.isfinite(score) or score < 0:
                raise ValueError('Invalid seed score')
            incoming.append((self.cfg.requested(seed['x'], seed['y']), score))
        new_segment = segment != self.segment
        count = 1 if new_segment else self.count+1
        history = ([] if new_segment else self.history)[-(self.cfg.window-1):]+[timestamp_ns]
        due = count >= self.cfg.warmup+self.cfg.window and (count-self.cfg.warmup-self.cfg.window) % self.cfg.stride == 0
        if due and required_halo(history) > self.cfg.halo:
            raise ValueError('Timestamp span exceeds response-space halo; never silently clip')
        if new_segment:
            self.seeds.clear()
            self.visits = [-1]*self.cfg.tiles
        self.segment, self.last_index, self.last_timestamp = segment, index, timestamp_ns
        self.count, self.history = count, history
        self.seeds = {k: v for k, v in self.seeds.items() if index-v[0] < self.cfg.ttl}
        for tiles, score in incoming:
            for tile in tiles:
                old = self.seeds.get(tile)
                self.seeds[tile] = (index, max(score, old[1]) if old and old[0] == index else score)
        if not due:
            return None
        blind = sorted(range(self.cfg.tiles), key=lambda k: (self.visits[k], k))[:self.cfg.blind]
        selected = list(blind)
        priority = sorted(self.seeds, key=lambda k: (-self.seeds[k][0], -self.seeds[k][1], k))
        for tile in priority:
            if tile not in selected and len(selected) < self.cfg.cap:
                selected.append(tile)
        # Spare capacity goes to blind work; empty scenes still search all slots.
        for tile in sorted(range(self.cfg.tiles), key=lambda k: (self.visits[k], k)):
            if len(selected) == self.cfg.cap:
                break
            if tile not in selected:
                selected.append(tile)
        gaps = [index-self.visits[k] for k in selected if self.visits[k] >= 0]
        for tile in selected:
            self.visits[tile] = index
        def area(k, halo):
            x0, y0, x1, y1 = self.cfg.rect(k, halo)
            return (x1-x0)*(y1-y0)
        return dict(frame=index, timestamp_ns=timestamp_ns, segment=segment,
                    tiles=selected, blind_reserved=blind, requested=len(self.seeds),
                    requested_serviced=sum(k in self.seeds for k in selected),
                    requested_deferred=sum(k not in selected for k in self.seeds),
                    core_pixels=sum(area(k, False) for k in selected),
                    halo_pixels_including_overlap=sum(area(k, True) for k in selected),
                    revisit_gaps_frames=gaps,
                    unvisited_tiles=sum(v < 0 for v in self.visits),
                    oldest_visited_age_frames=max((index-v for v in self.visits if v >= 0), default=0))


def queue_model(service_ms, fps=10.):
    """Infinite-buffer FIFO on one serial worker: modeled, not live measurement."""
    if not service_ms or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Nonempty services and positive cadence required')
    if any(not math.isfinite(v) or v < 0 for v in service_ms):
        raise ValueError('Nonnegative finite service times required')
    finish, waits, latencies = 0., [], []
    period = 1000/fps
    for i, service in enumerate(service_ms):
        arrival = i*period
        start = max(arrival, finish)
        finish = start+service
        waits.append(start-arrival)
        latencies.append(finish-arrival)
    def percentile(v, q):
        return sorted(v)[max(0, math.ceil(q*len(v))-1)]
    return dict(modeled_not_live=True, fps=fps, frames=len(service_ms),
                mean_service_ms=sum(service_ms)/len(service_ms),
                p95_service_ms=percentile(service_ms, .95),
                mean_exceeds_budget=sum(service_ms)/len(service_ms) > period,
                final_wait_ms=waits[-1], maximum_wait_ms=max(waits),
                p95_arrival_to_completion_ms=percentile(latencies, .95),
                final_arrival_to_completion_ms=latencies[-1],
                dropped_frames=0, queue_policy='unbounded FIFO; intentionally exposes backlog')
