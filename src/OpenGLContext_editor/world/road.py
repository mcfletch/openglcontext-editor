"""Putting a road into a world.

The engine sweeps a cross-section along a centreline
(:mod:`OpenGLContext.scenegraph.road`). What is here is everything that decides
*where* the centreline goes and what the world does about it:

:class:`RoadPath`
    a centreline in the world, and the distance query everything else is built
    on -- how far a point is from the road, and how high the road is beside it
:func:`follow_terrain`
    a drawn line settled onto the ground and smoothed into a grade a car can
    actually drive
:func:`conform_terrain`
    a height function that meets the road: the ground takes the road's own
    cross-section out to the verge and blends back to natural beyond it
:class:`RoadLayer`
    the road as a bake layer, re-sampled coarser for distant tiles and clipped
    to each tile it crosses

This is the "highway on dirt" operation -- the road follows the ground and the
ground is reshaped to meet it. Causeways, bridges and tunnels are the other
three, and are chosen by how far the alignment departs from the terrain.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import ExternalImage, SceneNode
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.road import (
    RoadProfile,
    resample_polyline,
    road_mesh,
    road_texture,
    tarmac_material,
)

from OpenGLContext_editor.bake.bounds import BoundingBox

HeightFn = Callable[[Any, Any], Any]

#: How far the ground takes to return to its natural shape beyond the verge, in
#: metres. Short enough that a road does not flatten the landscape around it,
#: long enough that the join is not a step.
DEFAULT_BLEND = 12.0

#: Centreline spacing at the finest tile, in metres, and how much coarser it
#: gets per metre of a tile's geometric error. A distant tile spends a fraction
#: of the vertices on the same road.
FINEST_SPACING = 3.0
SPACING_PER_ERROR = 1.5

#: How far outside a tile the road is still looked for, as a multiple of the
#: road's half-width, when deciding whether the tile has any road in it at all.
#: A road that only clips a corner still has to be found there.
TILE_REACH = 2.0

#: The road surface, written once beside the tileset. Every tile the road
#: crosses names this file; embedding a copy in each of them would cost more
#: than all the geometry in the world put together.
SURFACE_IMAGE = 'road-surface.png'


class RoadPath:
    """A road's centreline through the world, and what it does to the ground.

    Built from (N,3) world points, already at the height the road runs at.
    Every query is vectorised over the points asked about and pre-filtered to
    the segments that could possibly matter, so sampling a whole terrain tile
    against a four-kilometre road costs the handful of segments crossing it.
    """

    def __init__(self, points: Any, profile: RoadProfile | None = None) -> None:
        line = np.asarray(points, dtype='d').reshape(-1, 3)
        if len(line) < 2:
            raise ValueError("a road needs a centreline of at least two points")
        self.points = line
        self.profile = profile or RoadProfile()
        self._start = line[:-1]
        self._delta = line[1:] - line[:-1]
        lengths = np.einsum('ij,ij->i', self._delta[:, [0, 2]], self._delta[:, [0, 2]])
        # A zero-length segment has no direction to project onto; keeping the
        # divisor at one leaves its projection pinned to its start point.
        self._length2 = np.where(lengths > 0, lengths, 1.0)
        self._low = np.minimum(self._start, line[1:])
        self._high = np.maximum(self._start, line[1:])
        runs = np.linalg.norm(self._delta[:, [0, 2]], axis=1)
        rises = np.abs(self._delta[:, 1])
        self.max_grade = float((rises / np.where(runs > 0, runs, 1.0)).max())

    @property
    def length(self) -> float:
        """How long the road is, following the ground."""
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def bounds(self) -> BoundingBox:
        """The centreline's box, widened by the road's own half-width."""
        half = self.profile.total_width / 2.0
        box = BoundingBox.of_points(self.points)
        assert box is not None                   # the constructor refuses an empty line
        return BoundingBox((box.minimum[0] - half, box.minimum[1] - half,
                            box.minimum[2] - half),
                           (box.maximum[0] + half, box.maximum[1] + half,
                            box.maximum[2] + half))

    def resampled(self, spacing: float) -> np.ndarray:
        """The centreline at an even spacing -- the road's level of detail."""
        line: np.ndarray = resample_polyline(self.points, spacing)
        return line

    def nearest(self, x: Any, z: Any, radius: float | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """Distance from the road, and the road's height, at each (x, z).

        Distance is measured in the ground plane to the nearest point of the
        centreline; the height is the centreline's at that same point, which is
        what the ground has to meet. Points further than ``radius`` from every
        segment get an infinite distance and a height of zero, so a caller that
        only cares about the road's neighbourhood pays for nothing else.
        """
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        # The answer comes back the shape of the question, so a scalar query
        # yields a scalar and a terrain grid yields a grid.
        shape = np.broadcast_shapes(x.shape, z.shape)
        flat_x, flat_z = np.broadcast_to(x, shape).ravel(), \
            np.broadcast_to(z, shape).ravel()
        distance = np.full(flat_x.shape, np.inf)
        height = np.zeros(flat_x.shape)

        segments = self._candidates(flat_x, flat_z, radius)
        if len(segments):
            start = self._start[segments]
            delta = self._delta[segments]
            length2 = self._length2[segments]
            # Project each query onto each candidate segment, clamped to it.
            offset_x = flat_x[:, None] - start[None, :, 0]
            offset_z = flat_z[:, None] - start[None, :, 2]
            t = np.clip((offset_x * delta[None, :, 0] + offset_z * delta[None, :, 2])
                        / length2[None, :], 0.0, 1.0)
            near_x = start[None, :, 0] + t * delta[None, :, 0]
            near_z = start[None, :, 2] + t * delta[None, :, 2]
            gaps = np.hypot(flat_x[:, None] - near_x, flat_z[:, None] - near_z)
            best = gaps.argmin(axis=1)
            rows = np.arange(len(flat_x))
            distance = gaps[rows, best]
            height = start[best, 1] + t[rows, best] * delta[best, 1]
        return distance.reshape(shape), height.reshape(shape)

    def _candidates(self, x: np.ndarray, z: np.ndarray,
                    radius: float | None) -> np.ndarray:
        """Segment indices whose box could hold the nearest point to any query."""
        if not len(x):                           # pragma: no cover - empty query
            return np.zeros(0, dtype=np.intp)
        reach = float(radius) if radius is not None else np.inf
        if not np.isfinite(reach):
            return np.arange(len(self._start))
        inside = ((self._low[:, 0] - reach <= x.max())
                  & (self._high[:, 0] + reach >= x.min())
                  & (self._low[:, 2] - reach <= z.max())
                  & (self._high[:, 2] + reach >= z.min()))
        candidates: np.ndarray = np.nonzero(inside)[0]
        return candidates

    def section_offset(self, distance: Any) -> np.ndarray:
        """How far below the centreline the road's surface is, at a distance out.

        The profile's own cross-section, interpolated, and held at the verge's
        value beyond the road's edge -- which is the height the ground has to
        arrive at for the two to meet.
        """
        section = self.profile.section()
        lateral, vertical = section[:, 0], section[:, 1]
        half = lateral.max()
        offset: np.ndarray = np.interp(
            np.minimum(np.abs(np.asarray(distance, 'd')), half),
            lateral[lateral >= 0], vertical[lateral >= 0])
        return offset

    def crosses(self, region: BoundingBox, margin: float = 0.0) -> bool:
        """Whether the road comes within ``margin`` of a region, in plan."""
        half = self.profile.total_width / 2.0 + margin
        return bool(np.any(
            (self._low[:, 0] - half <= region.maximum[0])
            & (self._high[:, 0] + half >= region.minimum[0])
            & (self._low[:, 2] - half <= region.maximum[2])
            & (self._high[:, 2] + half >= region.minimum[2])))


def follow_terrain(course: Any, height_fn: HeightFn, spacing: float = 5.0,
                   smoothing: float = 60.0, clearance: float = 0.0,
                   maximum_grade: float | None = None,
                   minimum_height: float | None = None,
                   closed: bool = False) -> np.ndarray:
    """A drawn line turned into an alignment a car can drive.

    ``course`` is the plan the designer drew -- (N,2) as XZ, or (N,3) whose
    height is ignored. It is re-sampled at ``spacing``, dropped onto the ground,
    and then *smoothed*: a road that followed every hummock would be undrivable
    and would look wrong, so the grade is averaged over ``smoothing`` metres.
    ``clearance`` lifts the finished alignment, and ``maximum_grade`` (as a
    fraction, 0.08 for one in twelve) caps how steeply it may climb.

    ``minimum_height`` is a floor the alignment may not go below -- a waterline,
    with freeboard. Where the ground dips under it the road is lifted onto fill
    and its approaches are *raised* to reach the new level rather than the
    crossing being dropped back into the water, which is how a causeway takes a
    road over a flooded valley. The grade limit is honoured either way.

    ``closed`` says the course is a circuit that returns to its start. The
    smoothing then wraps around the join and the last point is made to match
    the first exactly, so a lap has no step in it at the start line.
    """
    plan = np.asarray(course, dtype='d')
    if plan.shape[1] == 2:
        plan = np.stack([plan[:, 0], np.zeros(len(plan)), plan[:, 1]], axis=-1)
    if closed and not np.allclose(plan[0], plan[-1]):
        plan = np.vstack([plan, plan[:1]])
    line: np.ndarray = resample_polyline(plan, spacing)
    line[:, 1] = np.asarray(height_fn(line[:, 0], line[:, 2]), dtype='d')
    if smoothing > 0 and len(line) > 2:
        line[:, 1] = _smooth(line[:, 1], window=max(int(smoothing / spacing), 1),
                             closed=closed)
    if maximum_grade is not None:
        line[:, 1] = _limit_grade(line, maximum_grade)
    if minimum_height is not None:
        line[:, 1] = np.maximum(line[:, 1], minimum_height)
        if maximum_grade is not None:
            line[:, 1] = _ramp_up_to_grade(line, maximum_grade)
    line[:, 1] += clearance
    if closed:
        line[0] = line[-1] = (line[0] + line[-1]) * 0.5
    return line


def _smooth(values: np.ndarray, window: int, closed: bool = False) -> np.ndarray:
    """A moving average over an alignment's heights.

    Edge-padded, so the first and last points of an open road stay on the ground
    the designer put them on; wrap-padded for a circuit, so the smoothing runs
    through the join instead of flattening towards it from both sides.
    """
    if window <= 1:
        return values
    padding = window // 2
    padded = np.pad(values, padding, mode='wrap' if closed else 'edge')
    kernel = np.ones(2 * padding + 1) / (2 * padding + 1)
    smoothed: np.ndarray = np.convolve(padded, kernel, mode='valid')[:len(values)]
    return smoothed


def _limit_grade(line: np.ndarray, maximum: float) -> np.ndarray:
    """Cap the climb between consecutive points, forwards then backwards.

    One pass in each direction, because a peak too high for its approach is also
    too high for its descent, and clipping only forwards would leave a cliff at
    the far side.
    """
    heights = line[:, 1].copy()
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    for index in range(1, len(heights)):
        limit = maximum * steps[index - 1]
        heights[index] = np.clip(heights[index], heights[index - 1] - limit,
                                 heights[index - 1] + limit)
    for index in range(len(heights) - 2, -1, -1):
        limit = maximum * steps[index]
        heights[index] = np.clip(heights[index], heights[index + 1] - limit,
                                 heights[index + 1] + limit)
    return heights


def _ramp_up_to_grade(line: np.ndarray, maximum: float) -> np.ndarray:
    """Raise whatever is needed so no step exceeds the grade, lowering nothing.

    The companion to :func:`_limit_grade`, used after a floor has lifted part of
    an alignment: the approaches climb to meet the raised section instead of the
    section being dropped back down to them.
    """
    heights = line[:, 1].copy()
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    for index in range(1, len(heights)):
        heights[index] = max(heights[index],
                             heights[index - 1] - maximum * steps[index - 1])
    for index in range(len(heights) - 2, -1, -1):
        heights[index] = max(heights[index],
                             heights[index + 1] - maximum * steps[index])
    return heights


def conform_terrain(height_fn: HeightFn, path: RoadPath,
                    blend: float = DEFAULT_BLEND,
                    widening: float = 0.0) -> HeightFn:
    """A height function whose ground meets the road.

    Out to the edge of the verge the ground *is* the road's cross-section, so
    the two surfaces coincide instead of one poking through the other. Beyond
    that the ground returns to its natural shape over ``blend`` metres, on a
    smoothstep, so there is no ridge where the earthwork ends.

    ``widening`` is how far apart the samples are that will read this function.
    A ground mesh only knows the surface at its vertices and draws straight
    lines between them, so a cutting narrower than that spacing is stepped over
    and the road inside it is buried by the ground it was cut into. Widening the
    earthwork by the spacing puts a sample inside the corridor whatever the
    resolution, which is why a coarse tile shows a broader, shallower cutting
    rather than none at all. Zero gives the earthwork as designed, which is what
    a collider, a scatter or a car asking where the ground is should use.

    The result is an ordinary height function: everything that samples terrain
    picks up the road's earthworks by being pointed at this instead.
    """
    half = path.profile.total_width / 2.0
    reach = half + blend + widening

    def conformed(x: Any, z: Any) -> np.ndarray:
        natural = np.asarray(height_fn(x, z), dtype='d')
        distance, road_height = path.nearest(x, z, radius=reach)
        distance = distance.reshape(natural.shape)
        # Widening pushes the *blend* outwards without touching the road's own
        # cross-section: ground beyond the verge is brought down to the verge's
        # level, never up to the crown's, or it would bury the shoulder it is
        # supposed to meet.
        outside = distance > half
        distance = np.where(outside, np.maximum(distance - widening, half), distance)
        road_height = road_height.reshape(natural.shape)
        near = distance < half + blend
        if not np.any(near):
            return natural
        # Beside a climbing road, the ground has to sit low enough that the
        # straight line a mesh draws between two samples cannot cross the
        # carriageway between them. The most the road can rise over one spacing
        # is its steepest grade, so that is how much further down the earthwork
        # goes at that resolution.
        sag = np.where(outside, widening * path.max_grade, 0.0)
        edge = road_height + path.section_offset(distance) - sag
        away = np.clip((distance - half) / blend, 0.0, 1.0)
        weight = away * away * (3.0 - 2.0 * away)      # smoothstep
        blended = edge * (1.0 - weight) + natural * weight
        return np.where(near, blended, natural)

    return conformed


def conform_terrain_at(height_fn: HeightFn, path: RoadPath,
                       blend: float = DEFAULT_BLEND
                       ) -> Callable[[float], HeightFn]:
    """The conformed ground as a function of the spacing it will be sampled at.

    Hand this to a :class:`~OpenGLContext_editor.bake.layers.HeightfieldLayer`
    as its ``height_fn_at``: every tile then gets the earthwork widened to its
    own resolution, so the road is legible at every level of the tree instead of
    only at the finest.
    """
    def at(spacing: float) -> HeightFn:
        return conform_terrain(height_fn, path, blend=blend, widening=spacing)
    return at


@dataclass
class RoadLayer:
    """A road as a bake layer: written into every tile it crosses.

    The centreline is re-sampled by the tile's geometric error, so a distant
    tile carries the same road at a fraction of the vertices, and clipped to the
    tile with an overlap so consecutive tiles join without a gap.
    """

    path: RoadPath
    material: PBRMaterial | None = None
    finest_spacing: float = FINEST_SPACING
    spacing_per_error: float = SPACING_PER_ERROR
    wetness: float = 0.0
    texture_size: int = 512
    seed: int = 0
    name: str = 'road'
    _material: PBRMaterial = field(init=False, repr=False)
    _surface: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.material is not None:
            self._material = self.material
            return
        # The surface is written beside the tileset and named by every tile,
        # so the material points at a file rather than carrying the pixels.
        self._surface = road_texture(self.texture_size, self.seed)
        self._material = tarmac_material(
            wetness=self.wetness,
            image=ExternalImage(SURFACE_IMAGE, srgb=True))

    def assets(self) -> dict[str, bytes]:
        """The road surface, for the bake to write once beside the tileset."""
        if self._surface is None:
            return {}
        buffer = io.BytesIO()
        self._surface.save(buffer, format='PNG')
        return {SURFACE_IMAGE: buffer.getvalue()}

    def bounds(self) -> BoundingBox:
        return self.path.bounds()

    def spacing_for(self, error: float) -> float:
        """How far apart the centreline points are for a tile of this error."""
        return max(self.finest_spacing, float(error) * self.spacing_per_error)

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        reach = self.path.profile.total_width * TILE_REACH
        if not self.path.crosses(region, margin=reach):
            return []
        line = self.path.resampled(self.spacing_for(error))
        return [SceneNode(mesh=road_mesh(run, self.path.profile,
                                         material=self._material),
                          name=self.name)
                for run in _runs_inside(line, region)]


def _runs_inside(line: np.ndarray, region: BoundingBox) -> list[np.ndarray]:
    """The stretches of a centreline a tile is responsible for drawing.

    A road can enter and leave one tile several times, so this returns a run per
    crossing rather than one clipped line.

    Each run reaches back one point before it enters the tile, and no further
    forward than the last point inside. That makes every segment of the road the
    responsibility of exactly one tile -- the one holding its far end -- so the
    surface is continuous across a tile boundary, where two neighbours share a
    vertex ring, and no stretch is drawn twice. Overlapping copies of a road
    would be worse than a gap: two coplanar surfaces fight for the depth buffer
    and the join shows as a band flickering across the carriageway.
    """
    # Half-open on the far side, so a point landing exactly on a boundary
    # belongs to one of the two neighbours sharing it rather than to both. The
    # outermost tile closes its far side again, or the last point of the road --
    # which can sit exactly on the world's edge -- would belong to no tile at
    # all and its final segment would go unwritten.
    inside = np.ones(len(line), dtype=bool)
    for axis in (0, 2):
        inside &= line[:, axis] >= region.minimum[axis]
        if region.maximum[axis] >= line[:, axis].max():
            inside &= line[:, axis] <= region.maximum[axis]
        else:
            inside &= line[:, axis] < region.maximum[axis]
    runs: list[tuple[int, int]] = []
    start = None
    for index, within in enumerate(inside):
        if within and start is None:
            start = index
        elif not within and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(inside)))
    out = []
    for first, last in runs:
        first = max(first - 1, 0)
        if last - first >= 2:
            out.append(line[first:last])
    return out
