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
    cross-section out to the verge and runs an earthwork out to meet the land
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

#: Below this many samples a query is answered in one go: grouping them by
#: cell costs a sort, and a handful of points is not worth sorting.
INDEX_THRESHOLD = 256

#: How big a cell of the query index is, as a fraction of the reach being
#: asked about. Smaller cells look at less road each and cost more passes to
#: do it; half a reach is where the two stop paying for each other.
INDEX_CELL = 0.5

#: How steeply an earthwork may fall away from the road, as a fraction: about
#: one in one and two-thirds, near the steepest earth stands at unheld. It is
#: what decides how wide an embankment or a cutting is, since the batter runs
#: from the shoulder until it meets the land.
EARTHWORK_SLOPE = 0.6

#: How far out from the verge the machine will go, in metres. A departure from
#: the ground the batter cannot close within this is left alone -- that is where
#: a bridge or a tunnel belongs.
MAXIMUM_EARTHWORK = 250.0

#: How far below the road's surface the ground under it sits, in metres. A road
#: is built on a formation and surfaced on top of it, so the ground beneath is
#: not the tarmac; and two surfaces at exactly the same height fight over which
#: one is drawn, which shows as the ground flickering through the carriageway
#: along the grid the terrain is sampled on.
FORMATION_DEPTH = 0.12

#: Centreline spacing at the finest tile, in metres, and how much coarser it
#: gets per metre of a tile's geometric error. A distant tile spends a fraction
#: of the vertices on the same road.
FINEST_SPACING = 3.0
SPACING_PER_ERROR = 1.5

#: How far outside a tile the road is still looked for, as a multiple of the
#: road's half-width, when deciding whether the tile has any road in it at all.
#: A road that only clips a corner still has to be found there.
TILE_REACH = 2.0

#: Standard gravity, m/s**2 -- what a car has available to hold it down over a
#: crest, and so what sets how sharp a crest may be.
GRAVITY = 9.81

#: How much of a car's weight a crest may take off its wheels. A car that keeps
#: three-quarters of its weight on the road over a crest still steers and still
#: brakes; one that keeps none of it is a car in the air.
CREST_WEIGHT_LOSS = 0.25

#: How many times the grade limit and the curvature limit are applied in turn.
#: Each is a projection onto a set the other can leave, and both sets contain a
#: level road, so alternating them converges on an alignment inside both; a
#: dozen rounds settles the profiles a landscape produces.
SETTLING_ROUNDS = 12

#: How the curvature limit is relaxed: how much of each point's correction is
#: taken per sweep, how many sweeps it may take, and the movement below which
#: the profile counts as settled (metres). Half of the correction converges
#: without the ringing a full step gives a run of adjacent points.
CURVATURE_RELAXATION = 0.5
CURVATURE_SWEEPS = 4000
CURVATURE_TOLERANCE = 1e-9

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
        for rows, segments in self._batches(flat_x, flat_z, radius):
            distance[rows], height[rows] = self._measure(
                flat_x[rows], flat_z[rows], segments)
        return distance.reshape(shape), height.reshape(shape)

    def comparisons(self, x: Any, z: Any, radius: float | None = None) -> int:
        """How many sample-against-segment comparisons a query would cost.

        The work, rather than the clock: what the index is for is that a whole
        landscape does not compare every ground sample against every segment of
        a four-kilometre road, and that is a thing to be able to measure.
        """
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        shape = np.broadcast_shapes(x.shape, z.shape)
        flat_x, flat_z = np.broadcast_to(x, shape).ravel(), \
            np.broadcast_to(z, shape).ravel()
        return sum(len(rows) * len(segments)
                   for rows, segments in self._batches(flat_x, flat_z, radius))

    def _measure(self, x: np.ndarray, z: np.ndarray, segments: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Distance and road height for a batch, against these segments."""
        start = self._start[segments]
        delta = self._delta[segments]
        length2 = self._length2[segments]
        # Project each query onto each candidate segment, clamped to it.
        offset_x = x[:, None] - start[None, :, 0]
        offset_z = z[:, None] - start[None, :, 2]
        t = np.clip((offset_x * delta[None, :, 0] + offset_z * delta[None, :, 2])
                    / length2[None, :], 0.0, 1.0)
        near_x = start[None, :, 0] + t * delta[None, :, 0]
        near_z = start[None, :, 2] + t * delta[None, :, 2]
        gaps = np.hypot(x[:, None] - near_x, z[:, None] - near_z)
        best = gaps.argmin(axis=1)
        rows = np.arange(len(x))
        return gaps[rows, best], start[best, 1] + t[rows, best] * delta[best, 1]

    def _batches(self, x: np.ndarray, z: np.ndarray, radius: float | None
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(query rows, segment indices)`` pairs covering the whole question.

        Queries are grouped by which cell of a coarse grid they fall in, and
        each cell is answered against the segments that come within ``radius``
        of it. A landscape's samples arrive in a grid, so the grouping costs one
        pass and the saving is the whole of the road it does not look at; a cell
        the road comes nowhere near is answered with nothing at all.
        """
        if not len(x):                           # pragma: no cover - empty query
            return []
        reach = float(radius) if radius is not None else np.inf
        if not np.isfinite(reach) or len(x) < INDEX_THRESHOLD:
            segments = self._near(x, z, reach)
            return [(np.arange(len(x)), segments)] if len(segments) else []
        cell = max(reach * INDEX_CELL, 1e-6)
        keys = (np.floor(x / cell).astype(np.int64) << 32) \
            + np.floor(z / cell).astype(np.int64)
        order = np.argsort(keys, kind='stable')
        edges = np.flatnonzero(np.diff(keys[order])) + 1
        batches = []
        for rows in np.split(order, edges):
            segments = self._near(x[rows], z[rows], reach)
            if len(segments):
                batches.append((rows, segments))
        return batches

    def _near(self, x: np.ndarray, z: np.ndarray, reach: float) -> np.ndarray:
        """Segment indices whose box comes within ``reach`` of these queries."""
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
                   design_speed: float | None = None,
                   closed: bool = False) -> np.ndarray:
    """A drawn line turned into an alignment a car can drive.

    ``course`` is the plan the designer drew -- (N,2) as XZ, or (N,3) whose
    height is ignored. It is re-sampled at ``spacing``, dropped onto the ground,
    and then *smoothed*: a road that followed every hummock would be undrivable
    and would look wrong, so the grade is averaged over ``smoothing`` metres.
    ``clearance`` lifts the finished alignment, and ``maximum_grade`` (as a
    fraction, 0.08 for one in twelve) caps how steeply it may climb.

    ``design_speed``, in metres per second, is how fast the road is meant to be
    driven, and it rounds off the changes of grade: a road that goes from
    climbing at its limit to descending at it inside a few metres is a ramp, and
    a car meeting it at speed leaves the ground because there is nothing under
    it. See :func:`curvature_limit` for what the speed buys. Left out, the
    alignment keeps whatever crests the landscape and the grade limit give it.

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
    bend = (curvature_limit(design_speed) if design_speed is not None
            else float('inf'))
    if maximum_grade is not None or np.isfinite(bend):
        line[:, 1] = _settle_profile(line, maximum_grade, bend, closed)
    if minimum_height is not None:
        line[:, 1] = np.maximum(line[:, 1], minimum_height)
        if maximum_grade is not None:
            line[:, 1] = _ramp_up_to_grade(line, maximum_grade, closed=closed)
        if np.isfinite(bend):
            # The floor put corners back into the profile where it lifted the
            # road; round them off again, keeping the road above the water.
            floor = line[:, 1].copy()
            line[:, 1] = _settle_profile(line, maximum_grade, bend, closed)
            line[:, 1] = np.maximum(line[:, 1], floor.min() * 0.0 + minimum_height)
    line[:, 1] += clearance
    if closed:
        # The ends are the same place, so give them the same height rather than
        # two that differ by a rounding step.
        line[0, 1] = line[-1, 1] = (line[0, 1] + line[-1, 1]) * 0.5
    return line


def curvature_limit(design_speed: float,
                    weight_loss: float = CREST_WEIGHT_LOSS) -> float:
    """The sharpest crest a road may have, as change of grade per metre.

    A vertical curve of radius ``R`` taken at ``v`` pulls a car off the road at
    ``v**2 / R``; keeping ``weight_loss`` of its weight or less off the wheels
    needs ``R >= v**2 / (g * weight_loss)``. Change of grade per metre is the
    reciprocal of that radius, which is the form the alignment is checked in.

    ``design_speed`` is in metres per second. A road with no design speed has no
    limit -- there is always some speed at which any crest launches a car, and
    the answer to that is to say how fast the road is meant to be driven.
    """
    if design_speed <= 0 or weight_loss <= 0:
        return float('inf')
    return GRAVITY * weight_loss / (design_speed * design_speed)


def _limit_curvature(line: np.ndarray, maximum: float,
                     closed: bool = False) -> np.ndarray:
    """Round off the changes of grade to ``maximum`` per metre.

    Wherever three consecutive points bend more sharply than that, the middle
    one moves towards the chord: by half of the least movement that would remove
    the excess outright, so a run of sharp points settles together instead of
    each overshooting for the next to undo. Repeated until nothing needs to
    move, which spreads a kink into the vertical curve that replaces it. An open
    road's ends stay where the designer put them.
    """
    heights = np.asarray(line[:, 1], dtype='d').copy()
    count = len(heights)
    if count < 3 or not np.isfinite(maximum):
        return heights
    steps = _steps(line, closed)
    if closed:
        forward = np.asarray(steps, dtype='d')
        back = np.roll(forward, 1)
    else:
        # The end steps are never used: the ends do not move.
        forward = np.concatenate([steps, steps[-1:]])
        back = np.concatenate([steps[:1], steps])
    span = 0.5 * (back + forward)
    lever = back * forward / (back + forward)
    for _sweep in range(CURVATURE_SWEEPS):
        bend = ((np.roll(heights, -1) - heights) / forward
                - (heights - np.roll(heights, 1)) / back)
        excess = np.abs(bend) - maximum * span
        shift = np.where(excess > 0.0,
                         np.sign(bend) * excess * CURVATURE_RELAXATION * lever,
                         0.0)
        if not closed:
            shift[0] = shift[-1] = 0.0
        if float(np.abs(shift).max()) < CURVATURE_TOLERANCE:
            break
        heights += shift
    return heights


def _settle_profile(line: np.ndarray, maximum_grade: float | None,
                    maximum_curvature: float, closed: bool) -> np.ndarray:
    """Bring an alignment inside the grade *and* the curvature limit.

    Rounding a crest off can steepen what leads to it, and clipping a grade puts
    a corner back into the profile, so neither limit can be applied once and
    left. Each is a projection onto a set of alignments -- both of them convex,
    both of them containing a level road -- so applying them in turn converges
    on one inside both. The curvature goes last, because a road half a percent
    too steep is still a road and a road with a kink in it is a ramp.
    """
    heights = line[:, 1]
    if not np.isfinite(maximum_curvature):
        return (_limit_grade(line, maximum_grade, closed=closed)
                if maximum_grade is not None else heights)
    # A closed course arrives with its first point repeated at the end, so the
    # two are one point with a step of nothing between them. Relaxing the bend
    # at a zero-length step is meaningless; the loop is settled over the points
    # it actually has, and the repeat takes the answer at the end.
    repeated = closed and bool(np.allclose(line[0, [0, 2]], line[-1, [0, 2]]))
    working = (line[:-1] if repeated else line).copy()
    for _round in range(SETTLING_ROUNDS):
        if maximum_grade is not None:
            working[:, 1] = _limit_grade(working, maximum_grade, closed=closed)
        working[:, 1] = _limit_curvature(working, maximum_curvature, closed=closed)
    if repeated:
        return np.concatenate([working[:, 1], working[:1, 1]])
    return working[:, 1]


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


def _limit_grade(line: np.ndarray, maximum: float,
                 closed: bool = False) -> np.ndarray:
    """Cap the climb between consecutive points, forwards then backwards.

    One pass in each direction, because a peak too high for its approach is also
    too high for its descent, and clipping only forwards would leave a cliff at
    the far side.

    ``closed`` runs each pass twice round the loop rather than once along it. A
    circuit's steepest place is as likely to be at the join as anywhere else,
    and a limit that stopped at the ends would leave the start line at the top
    of a cliff -- which is exactly where a car is put.
    """
    heights = line[:, 1].copy()
    steps = _steps(line, closed)
    count = len(heights)
    for index in _forward(count, closed):
        behind = (index - 1) % count
        limit = maximum * steps[behind]
        heights[index] = np.clip(heights[index], heights[behind] - limit,
                                 heights[behind] + limit)
    for index in _backward(count, closed):
        ahead = (index + 1) % count
        limit = maximum * steps[index]
        heights[index] = np.clip(heights[index], heights[ahead] - limit,
                                 heights[ahead] + limit)
    return heights


def _steps(line: np.ndarray, closed: bool) -> np.ndarray:
    """Ground distance between consecutive points, wrapping for a circuit."""
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    if closed:
        # One more entry, from the last point back to the first, so index -1
        # reaches it and the wrap has a length to work with.
        joining = float(np.linalg.norm(line[0, [0, 2]] - line[-1, [0, 2]]))
        steps = np.concatenate([steps, [max(joining, 1e-6)]])
    return steps


def _forward(count: int, closed: bool) -> list[int]:
    """Indices for a forward pass; twice round for a closed loop."""
    if not closed:
        return list(range(1, count))
    return [index % count for index in range(1, 2 * count + 1)]


def _backward(count: int, closed: bool) -> list[int]:
    """Indices for a backward pass; twice round for a closed loop."""
    if not closed:
        return list(range(count - 2, -1, -1))
    return [index % count for index in range(2 * count - 2, -2, -1)]


def _ramp_up_to_grade(line: np.ndarray, maximum: float,
                      closed: bool = False) -> np.ndarray:
    """Raise whatever is needed so no step exceeds the grade, lowering nothing.

    The companion to :func:`_limit_grade`, used after a floor has lifted part of
    an alignment: the approaches climb to meet the raised section instead of the
    section being dropped back down to them.
    """
    heights = line[:, 1].copy()
    steps = _steps(line, closed)
    count = len(heights)
    for index in _forward(count, closed):
        behind = (index - 1) % count
        heights[index] = max(heights[index],
                             heights[behind] - maximum * steps[behind])
    for index in _backward(count, closed):
        ahead = (index + 1) % count
        heights[index] = max(heights[index],
                             heights[ahead] - maximum * steps[index])
    return heights


def conform_terrain(height_fn: HeightFn, path: RoadPath,
                    earthwork_slope: float = EARTHWORK_SLOPE,
                    maximum_earthwork: float = MAXIMUM_EARTHWORK,
                    formation: float = FORMATION_DEPTH,
                    widening: float = 0.0) -> HeightFn:
    """A height function whose ground meets the road.

    Out to the edge of the verge the ground *is* the road's cross-section, so
    the two surfaces coincide instead of one poking through the other. Beyond
    it the ground is an *earthwork*: fill runs down from the shoulder to where
    it meets the land, a cutting runs up to it, and how far out that is depends
    on how far the road is from the ground and on nothing else. A road already
    on the ground disturbs almost nothing; one carried forty metres over a
    valley builds an embankment as wide as it needs.

    ``formation`` is how far below the road's surface the ground under it is
    set: a road is built on a formation and surfaced on top of it, and two
    surfaces at the same height fight over which one is drawn.

    ``earthwork_slope`` is how steeply that batter may fall, as a fraction:
    0.6 is about one in one and two-thirds, near the steepest earth stands at
    unheld. ``maximum_earthwork`` is how far out the machine will go; a
    departure it cannot reach the ground within is left as it is, because that
    is where a bridge or a tunnel belongs and moving a mountain would only hide
    the fact.

    ``widening`` is how far apart the samples are that will read this function.
    A ground mesh only knows the surface at its vertices and draws straight
    lines between them, so a cutting narrower than that spacing is stepped over
    and the road inside it is buried by the ground it was cut into. Holding a
    shelf at the verge for that spacing before the batter starts puts a sample
    inside the corridor whatever the resolution, which is why a coarse tile
    shows a broader, shallower cutting rather than none at all. Zero gives the
    earthwork as designed, which is what a collider, a scatter or a car asking
    where the ground is should use.

    The result is an ordinary height function: everything that samples terrain
    picks up the road's earthworks by being pointed at this instead.
    """
    half = path.profile.total_width / 2.0
    reach = half + widening + maximum_earthwork

    def conformed(x: Any, z: Any) -> np.ndarray:
        natural = np.asarray(height_fn(x, z), dtype='d')
        distance, road_height = path.nearest(x, z, radius=reach)
        distance = distance.reshape(natural.shape)
        road_height = road_height.reshape(natural.shape)
        near = distance < reach
        if not np.any(near):
            return natural
        # Beside a climbing road, the ground has to sit low enough that the
        # straight line a mesh draws between two samples cannot cross the
        # carriageway between them. The most the road can rise over one spacing
        # is its steepest grade, so that is how much further down the earthwork
        # goes at that resolution.
        outside = distance > half
        sag = np.where(outside, widening * path.max_grade, 0.0)
        section = (road_height + path.section_offset(np.minimum(distance, half))
                   - sag - formation)
        # Out to the widening the ground is held at the verge; past that the
        # batter falls away at its slope until it reaches the land.
        beyond = np.clip(distance - half - widening, 0.0, maximum_earthwork)
        room = earthwork_slope * beyond
        earthwork = np.clip(natural, section - room, section + room)
        return np.where(near, np.where(outside, earthwork, section), natural)

    return conformed


def conform_terrain_at(height_fn: HeightFn, path: RoadPath,
                       earthwork_slope: float = EARTHWORK_SLOPE,
                       maximum_earthwork: float = MAXIMUM_EARTHWORK,
                       formation: float = FORMATION_DEPTH,
                       ) -> Callable[[float], HeightFn]:
    """The conformed ground as a function of the spacing it will be sampled at.

    Hand this to a :class:`~OpenGLContext_editor.bake.layers.HeightfieldLayer`
    as its ``height_fn_at``: every tile then gets the earthwork widened to its
    own resolution, so the road is legible at every level of the tree instead of
    only at the finest.
    """
    def at(spacing: float) -> HeightFn:
        return conform_terrain(height_fn, path, earthwork_slope=earthwork_slope,
                               maximum_earthwork=maximum_earthwork,
                               formation=formation, widening=spacing)
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
    metadata_spacing: float = 8.0
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

    def metadata(self) -> dict[str, Any]:
        """The road itself, for the tileset's extras.

        A game cannot find a road in a pile of triangles, and needs it to put a
        car on the track, time a lap and drive an opponent round. So the road
        travels with the world it is baked into: its centreline, how wide it is,
        and whether it closes into a circuit.

        The line is written at ``metadata_spacing`` rather than at its full
        density -- a course is a shape, and a game re-samples it for whatever it
        is doing.
        """
        line = self.path.resampled(self.metadata_spacing)
        closed = bool(np.allclose(self.path.points[0], self.path.points[-1]))
        return {'roads': [{
            'name': self.name,
            'closed': closed,
            'carriagewayWidth': self.path.profile.carriageway_width,
            'totalWidth': self.path.profile.total_width,
            'length': self.path.length,
            'centreline': [[round(float(v), 3) for v in point] for point in line],
        }]}

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
