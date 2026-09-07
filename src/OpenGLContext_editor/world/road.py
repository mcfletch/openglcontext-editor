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
    to each tile it crosses, with the deck or the bore where the road is on one

A road that has been through
:func:`~OpenGLContext_editor.world.structures.choose_structures` arrives here
knowing which of its stretches stand on the land and which are carried over or
through it. Where it stands on the land the ground is reshaped to meet it --
the "highway on dirt" operation, with a causeway as the case where the fill
carries it over drowned ground. Where it does not, the terrain is left as it
was and a structure is built instead.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from OpenGLContext.loaders.gltf.writer import ExternalImage, SceneNode
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.road import (
    RoadProfile,
    banked_sections,
    morphed_sections,
    resample_polyline,
    road_mesh,
    road_texture,
    tarmac_material,
    widened_sections,
)
from OpenGLContext.scenegraph.roadworks import (
    BridgeProfile,
    CausewayProfile,
    TunnelProfile,
    barrier_material,
    bridge_meshes,
    causeway_meshes,
    concrete_material,
    tunnel_lamps,
    tunnel_meshes,
)

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.structures import Op

HeightFn = Callable[[Any, Any], Any]

#: The operations under which the road does not stand on the land, so the
#: terrain beneath (or above) it is left as it was found.
CARRIED = frozenset((Op.BRIDGE, Op.CAUSEWAY, Op.TUNNEL))


class RoadSample(NamedTuple):
    """What a road is doing at some point on the ground beside it.

    ``distance`` is how far that point is from the centreline in plan,
    ``height`` how high the road is at the nearest place on it, and ``segment``
    which segment of the centreline that place is on -- which is how a caller
    finds out what is built there.

    ``side`` is which hand of the road the point is on, +1 for its right and -1
    for its left, and ``bank`` how far the road leans at that place. The two go
    together: a level road's two verges are at one height and a banked one's are
    not, so anything meeting the road out at a distance has to know which side
    of it that distance is on.
    """

    distance: np.ndarray
    height: np.ndarray
    segment: np.ndarray
    side: np.ndarray
    bank: np.ndarray


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

#: Over how many metres the carriageway's cut changes onto a structure's. Long
#: enough that the verge flattens rather than steps, short enough that the
#: change happens at the abutment rather than half way down the approach.
TRANSITION_LENGTH = 24.0


def _bank_array(bank: Any, count: int) -> np.ndarray:
    """How far the road leans at each point, defaulting to not at all."""
    if bank is None:
        return np.zeros(count)
    found = np.asarray(bank, dtype='d').reshape(-1)
    if len(found) != count:
        raise ValueError(
            "a road of %d points needs %d leans, not %d"
            % (count, count, len(found)))
    return found


def _ops_array(ops: Any, count: int) -> np.ndarray:
    """What is built at each centreline point, defaulting to plain dirt."""
    if ops is None:
        return np.full(count, Op.DIRT, dtype=object)
    found = np.asarray(list(ops), dtype=object)
    if len(found) != count:
        raise ValueError(
            "a road of %d points needs %d operations, not %d"
            % (count, count, len(found)))
    return found


class RoadPath:
    """A road's centreline through the world, and what it does to the ground.

    Built from (N,3) world points, already at the height the road runs at.
    Every query is vectorised over the points asked about and pre-filtered to
    the segments that could possibly matter, so sampling a whole terrain tile
    against a four-kilometre road costs the handful of segments crossing it.
    """

    def __init__(self, points: Any, profile: RoadProfile | None = None,
                 ops: Any = None, bank: Any = None,
                 clearance: Any = None, widening: Any = None) -> None:
        line = np.asarray(points, dtype='d').reshape(-1, 3)
        if len(line) < 2:
            raise ValueError("a road needs a centreline of at least two points")
        self.points = line
        self.profile = profile or RoadProfile()
        #: How far the road leans at each point of the centreline, as a
        #: fraction and signed the way
        #: :func:`~OpenGLContext.scenegraph.road.plan_curvature` is. Zero from
        #: end to end for a road whose corners are flat, which is what a road
        #: nobody laid out for speed has.
        self.bank: np.ndarray = _bank_array(bank, len(line))
        #: Whether any of it leans at all. A flat road costs nothing extra
        #: anywhere the lean would otherwise have to be carried.
        self.banked = bool(np.any(self.bank))
        #: How far from the centreline the ground beside the road is cleared,
        #: at each point, in metres -- the corridor the road is built inside,
        #: opened out where a driver has to see round a bend
        #: (:func:`~OpenGLContext_editor.world.character.road_character`). None
        #: for a road whose corridor is the same width from end to end, which
        #: is what a caller that never asked for one has.
        self.clearance: np.ndarray | None = (
            None if clearance is None
            else _bank_array(clearance, len(line)))
        #: How much more carriageway the road has at each point, in metres --
        #: zero for its own width, a lane's worth on a stretch built to be
        #: passed on.
        self.widening: np.ndarray = _bank_array(widening, len(line))
        #: Whether any of it is wider than the road it is on.
        self.wider = bool(np.any(self.widening))
        #: What is built at each point of the centreline. A road that has not
        #: been through :func:`~OpenGLContext_editor.world.structures.choose_structures`
        #: is on dirt from end to end, which is the ordinary case and the one
        #: everything else falls back to.
        self.ops: np.ndarray = _ops_array(ops, len(line))
        #: Whether the road at each point stands on the land. A causeway does --
        #: it is fill, and the ground is what it is made of -- but a deck stands
        #: over the land and a bore inside it, and the terrain under or above
        #: those is left as it was.
        self.on_ground: np.ndarray = np.array(
            [op not in CARRIED for op in self.ops], dtype=bool)
        #: A segment is reshaped if *either* of its ends is laid on the land.
        #: The last stretch before a portal is on the ground and the ground has
        #: to come down to meet it; stopping one segment short leaves a lip of
        #: hillside across the road at the place a car arrives at speed.
        self.segment_on_ground = self.on_ground[:-1] | self.on_ground[1:]
        self._start = line[:-1]
        self._delta = line[1:] - line[:-1]
        lengths = np.einsum('ij,ij->i', self._delta[:, [0, 2]], self._delta[:, [0, 2]])
        # A zero-length segment has no direction to project onto; keeping the
        # divisor at one leaves its projection pinned to its start point.
        self._length2 = np.where(lengths > 0, lengths, 1.0)
        self._low = np.minimum(self._start, line[1:])
        self._high = np.maximum(self._start, line[1:])
        #: Which cells of the query index the road can reach, by (cell, reach).
        #: A bake asks the same question of millions of points in a row.
        self._occupancy: dict[tuple[float, float], np.ndarray] = {}
        runs = np.linalg.norm(self._delta[:, [0, 2]], axis=1)
        rises = np.abs(self._delta[:, 1])
        self.max_grade = float((rises / np.where(runs > 0, runs, 1.0)).max())
        #: Distance along the road to each of its points. Everything that has
        #: to line up two samplings of the same road -- a coarsened centreline
        #: against the operations chosen on the full one -- meets here.
        self.stations = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(self._delta, axis=1))])

    @property
    def length(self) -> float:
        """How long the road is, following the ground."""
        return float(self.stations[-1])

    def reshaped_segments(self) -> np.ndarray:
        """Which segments the earthwork reshapes the ground along.

        The ones standing on the land, and **every one running through a bore**.

        The ground a tunnel passes through has to come out from under it. Left
        in, the hillside stands inside the tube: a ground mesh draws straight
        lines between its samples, so the line from the last cut sample outside
        the portal up to the untouched hill is a bank across the opening -- the
        road arrives at a wall with the arch in the air behind it, and a car
        only gets through because the collider has the bore taken out of it.
        Cutting a sample's worth inside the portal only moves the bank a few
        metres in, where it is a hillside seen through the mouth instead of the
        lining.

        A height field is a surface, so what "out from under it" can mean here
        is *down to the road*: the approach cutting carries on through the hill
        for the length of the bore, with the lining standing inside it. The
        hill above a long bore is opened into a broad cutting, which is the
        cost of drawing a tunnel in a surface rather than in a solid; what
        would keep the hill whole is a hole in the ground mesh with the bore's
        own outside plugging it, and that is a terrain feature rather than a
        road one.

        **Either end in a bore is enough.** A ground mesh draws straight lines
        between its samples, so what the opening looks like is decided by the
        sample *at the mouth* -- and the segment straddling a portal has one end
        outside it. Reshape only the segments with both ends inside and that
        sample keeps the hillside's own height: the cut starts a segment late,
        and the line from the hill down to it is a wall across the road with the
        arch standing in the air behind it.

        Not a deck: under one the same move would raise a pillar of ground to
        meet a road that is forty metres up.
        """
        on = self.segment_on_ground
        bore = np.array([op is Op.TUNNEL for op in self.ops], dtype=bool)
        found: np.ndarray = on | bore[:-1] | bore[1:]
        return found

    def ops_at(self, stations: Any) -> np.ndarray:
        """What is built at each of these distances along the road.

        A tile carries the road at its own spacing, and the operations were
        chosen on the full-density alignment; matching them up is a lookup by
        distance rather than by index. Each query takes the operation of the
        nearest point of the alignment behind it, so a structure's extent is
        the stretch it was chosen for and not a point more.
        """
        wanted = np.asarray(stations, dtype='d')
        at = np.clip(np.searchsorted(self.stations, wanted, side='right') - 1,
                     0, len(self.ops) - 1)
        found: np.ndarray = self.ops[at]
        return found

    def widening_at(self, stations: Any) -> np.ndarray:
        """How much wider the carriageway is, at each distance along the road.

        The companion to :meth:`bank_at`: a tile carries the road at its own
        spacing, and how wide it is was worked out on the full alignment.
        """
        return np.interp(np.asarray(stations, dtype='d'), self.stations,
                         self.widening)

    def bank_at(self, stations: Any) -> np.ndarray:
        """How far the road leans at each of these distances along it.

        The companion to :meth:`ops_at`, and there for the same reason: a tile
        carries the road at its own spacing and the lean was worked out on the
        full-density alignment. Interpolated rather than stepped, because a lean
        is a continuous thing and a coarse tile that took the nearest point
        behind it would carry a road that rolled in steps.
        """
        return np.interp(np.asarray(stations, dtype='d'), self.stations,
                         self.bank)

    def clearance_along(self) -> np.ndarray | None:
        """How wide the cleared corridor is beside each *segment*, in metres.

        The wider of the corridor at the segment's two ends, so a tree is kept
        out of a clearing that either end of the segment it is beside asks for
        rather than slipping into the join between them. None for a road whose
        corridor never changes width.
        """
        if self.clearance is None:
            return None
        return np.maximum(self.clearance[:-1], self.clearance[1:])

    def structure_runs(self) -> list[tuple[Op, float, float]]:
        """Every stretch that is not plain dirt, as ``(op, from, to)`` metres.

        The road's own summary of what is built along it: what a tileset carries
        for a game to read, and what the bake layer places geometry from.
        """
        out: list[tuple[Op, float, float]] = []
        start = 0
        for index in range(1, len(self.ops) + 1):
            if index < len(self.ops) and self.ops[index] is self.ops[start]:
                continue
            if self.ops[start] is not Op.DIRT:
                out.append((self.ops[start], float(self.stations[start]),
                            float(self.stations[index - 1])))
            start = index
        return out

    def bounds(self) -> BoundingBox:
        """The centreline's box, widened by the road's own half-width."""
        half = self.widest() / 2.0
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
        found = self.sample(x, z, radius)
        return found.distance, found.height

    def sample(self, x: Any, z: Any, radius: float | None = None
               ) -> RoadSample:
        """What the road is doing at each (x, z): how far, how high, and where.

        :meth:`nearest` without the segment, which is what a caller needs to
        know *which stretch* of the road answered -- and so whether that
        stretch is standing on the land or being carried over it.
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
        segment = np.zeros(flat_x.shape, dtype=np.intp)
        side = np.ones(flat_x.shape)
        bank = np.zeros(flat_x.shape)
        for rows, segments in self._batches(flat_x, flat_z, radius):
            (distance[rows], height[rows], segment[rows], side[rows],
             bank[rows]) = self._measure(flat_x[rows], flat_z[rows], segments)
        return RoadSample(distance.reshape(shape), height.reshape(shape),
                          segment.reshape(shape), side.reshape(shape),
                          bank.reshape(shape))

    def index_cells(self, x: Any, z: Any, radius: float | None = None) -> int:
        """How many groups the index splits a query into.

        The Python-level cost of asking, which the comparison count does not
        show: a cell the road comes nowhere near contributes no comparisons and
        still costs an iteration if it is visited at all.
        """
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        shape = np.broadcast_shapes(x.shape, z.shape)
        return len(self._batches(np.broadcast_to(x, shape).ravel(),
                                 np.broadcast_to(z, shape).ravel(), radius))

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
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]:
        """Distance, height, segment, side and lean for a batch, against these."""
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
        along = t[rows, best]
        chosen = segments[best]
        run = self._delta[chosen]
        # Which hand of the road the point is on: the road's right vector in
        # plan is the segment turned a quarter clockwise, so the sign of the
        # offset along it says which side. A point exactly on the centreline
        # takes the right, since it has to take one and the crown is at the
        # same height either way.
        away_x = x - (start[best, 0] + along * run[:, 0])
        away_z = z - (start[best, 2] + along * run[:, 2])
        side = np.where(run[:, 0] * away_z - run[:, 2] * away_x < 0.0, -1.0, 1.0)
        lean = self.bank[chosen] + along * (self.bank[chosen + 1]
                                            - self.bank[chosen])
        return (gaps[rows, best],
                start[best, 1] + along * delta[best, 1],
                chosen, side, lean)

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
        keys = _cell_keys(x, z, cell)
        # Everything the road comes nowhere near, dropped in one pass. Grouping
        # by cell makes each comparison cheap; it does not make an *empty* cell
        # cheap, and a two-thousand-pixel map at a five-metre cell is most of a
        # million of them. Iterating over those costs more than the comparisons
        # the index saved.
        near = np.isin(keys, self._occupied(cell, reach))
        if not near.any():
            return []
        rows_near = np.nonzero(near)[0]
        keys = keys[rows_near]
        order = rows_near[np.argsort(keys, kind='stable')]
        edges = np.flatnonzero(np.diff(np.sort(keys))) + 1
        batches = []
        for rows in np.split(order, edges):
            segments = self._near(x[rows], z[rows], reach)
            if len(segments):
                batches.append((rows, segments))
        return batches

    def _occupied(self, cell: float, reach: float) -> np.ndarray:
        """The keys of every cell the road can reach, sorted.

        A segment's box grown by ``reach`` covers a block of cells; the union of
        those blocks is where an answer can be anything but "nowhere near".
        Cached per (cell, reach), because a bake asks the same question of
        millions of points in a row.
        """
        found = self._occupancy.get((cell, reach))
        if found is None:
            spread = int(np.ceil(reach / cell)) + 1
            steps = np.arange(-spread, spread + 1)
            block_x, block_z = np.meshgrid(steps, steps)
            base_x = np.floor(self.points[:, 0] / cell).astype(np.int64)
            base_z = np.floor(self.points[:, 2] / cell).astype(np.int64)
            keys = ((base_x[:, None] + block_x.ravel()[None, :]) << 32) \
                + (base_z[:, None] + block_z.ravel()[None, :])
            found = np.unique(keys.ravel())
            self._occupancy[(cell, reach)] = found
        return found

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

    def section_offset(self, distance: Any, side: Any = 1.0,
                       bank: Any = 0.0) -> np.ndarray:
        """How far below the centreline the road's surface is, at a distance out.

        ``distance`` is measured in plan, ``side`` is which hand of the road it
        is on (+1 for the right) and ``bank`` how far the road leans there --
        all three as :meth:`sample` gives them, so a caller hands the answer
        back in.

        The cut across a road is a property of its section
        (:meth:`~OpenGLContext.scenegraph.road.RoadProfile.section_offset`) and
        not of the line it is swept along, so that is where the shape comes
        from. What is here is the lean, which the line *does* own: a banked
        road's two verges are as far apart in height as the road is wide, and
        the ground meeting one of them has to be told which.

        Past the road's own edge the answer is held at the verge, level road or
        banked, since past there the surface is the ground rather than the road.
        """
        distance = np.asarray(distance, dtype='d')
        if not self.banked:
            # A road with no lean in it anywhere is asked this of every ground
            # sample of every tile it crosses. The cut alone is the answer, and
            # the arithmetic that would resolve a lean is arithmetic over
            # millions of zeroes.
            found: np.ndarray = self.profile.section_offset(
                np.minimum(distance, self.profile.total_width / 2.0))
            return found
        lean = np.asarray(bank, dtype='d')
        # The pavement rotates about the crown, so a cut `across` wide along a
        # leaning surface is `across * upright` wide in plan.
        upright = 1.0 / np.hypot(1.0, lean)
        across = np.minimum(distance / upright,
                            self.profile.total_width / 2.0)
        leaning: np.ndarray = (
            self.profile.section_offset(across, lean) * upright
            - np.asarray(side, dtype='d') * across * lean * upright)
        return leaning

    def widest(self) -> float:
        """Across everything the road occupies at its widest, verge to verge.

        The profile's own width, plus whatever the widest stretch built to be
        passed on adds to it: a road that reaches past its nominal width is a
        road whose tile is a metre short of it at the edge.
        """
        return float(self.profile.total_width + self.widening.max())

    def crosses(self, region: BoundingBox, margin: float = 0.0) -> bool:
        """Whether the road comes within ``margin`` of a region, in plan."""
        half = self.widest() / 2.0 + margin
        return bool(np.any(
            (self._low[:, 0] - half <= region.maximum[0])
            & (self._high[:, 0] + half >= region.minimum[0])
            & (self._low[:, 2] - half <= region.maximum[2])
            & (self._high[:, 2] + half >= region.minimum[2])))


def follow_terrain(course: Any, height_fn: HeightFn, spacing: float = 5.0,
                   smoothing: Any = 60.0, clearance: float = 0.0,
                   maximum_grade: Any = None,
                   minimum_height: float | None = None,
                   design_speed: Any = None,
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

    **``smoothing``, ``maximum_grade`` and ``design_speed`` may each be one
    figure or one per re-sampled point**, which is what makes a road that is not
    the same road all the way round: a stretch left rough where the rest is
    ironed flat, a climb steeper than the road it is on, a slow corner whose
    crests are rounded for the speed anybody will actually take it at rather
    than for the speed of the straight before it. Each is still a *limit* over
    the whole road -- what varies is how tight it is where. A per-point figure
    has to be as long as the alignment ``spacing`` produces; see
    :func:`points_along` for how many that is.
    """
    plan = np.asarray(course, dtype='d')
    if plan.shape[1] == 2:
        plan = np.stack([plan[:, 0], np.zeros(len(plan)), plan[:, 1]], axis=-1)
    if closed and not np.allclose(plan[0], plan[-1]):
        plan = np.vstack([plan, plan[:1]])
    line: np.ndarray = resample_polyline(plan, spacing)
    line[:, 1] = np.asarray(height_fn(line[:, 0], line[:, 2]), dtype='d')
    window = _per_point(smoothing, len(line), 'smoothing') / max(spacing, 1e-9)
    if float(window.max()) > 1.0 and len(line) > 2:
        line[:, 1] = _smooth(line[:, 1], window, closed=closed)
    grade = (None if maximum_grade is None
             else _per_point(maximum_grade, len(line), 'grade limits'))
    bend = (float('inf') if design_speed is None else curvature_limit(
        _per_point(design_speed, len(line), 'design speeds')))
    if grade is not None or np.any(np.isfinite(bend)):
        line[:, 1] = _settle_profile(line, grade, bend, closed)
    if minimum_height is not None:
        line[:, 1] = np.maximum(line[:, 1], minimum_height)
        if grade is not None:
            line[:, 1] = _ramp_up_to_grade(line, grade, closed=closed)
        if np.any(np.isfinite(bend)):
            # The floor put corners back into the profile where it lifted the
            # road; round them off again, keeping the road above the water.
            line[:, 1] = _settle_profile(line, grade, bend, closed)
            line[:, 1] = np.maximum(line[:, 1], minimum_height)
    line[:, 1] += clearance
    if closed:
        # The ends are the same place, so give them the same height rather than
        # two that differ by a rounding step.
        line[0, 1] = line[-1, 1] = (line[0, 1] + line[-1, 1]) * 0.5
    return line


def points_along(course: Any, spacing: float, closed: bool = False) -> int:
    """How many points :func:`follow_terrain` will re-sample a plan into.

    A caller working out a limit *per point* -- a grade that varies, a design
    speed that varies -- needs to know how many points there will be before
    there is an alignment to count. This is that, from the same plan and the
    same spacing, so the two cannot disagree.
    """
    plan = np.asarray(course, dtype='d')
    if plan.shape[1] == 2:
        plan = np.stack([plan[:, 0], np.zeros(len(plan)), plan[:, 1]], axis=-1)
    if closed and not np.allclose(plan[0], plan[-1]):
        plan = np.vstack([plan, plan[:1]])
    return len(resample_polyline(plan, spacing))


def _per_point(value: Any, count: int, what: str) -> np.ndarray:
    """A limit as one figure per point, from one figure or a whole array."""
    found = np.asarray(value, dtype='d')
    if found.ndim == 0:
        return np.full(count, float(found))
    found = found.reshape(-1)
    if len(found) != count:
        raise ValueError("an alignment of %d points needs %d %s, not %d"
                         % (count, count, what, len(found)))
    return found


def curvature_limit(design_speed: Any,
                    weight_loss: float = CREST_WEIGHT_LOSS) -> float | np.ndarray:
    """The sharpest crest a road may have, as change of grade per metre.

    A vertical curve of radius ``R`` taken at ``v`` pulls a car off the road at
    ``v**2 / R``; keeping ``weight_loss`` of its weight or less off the wheels
    needs ``R >= v**2 / (g * weight_loss)``. Change of grade per metre is the
    reciprocal of that radius, which is the form the alignment is checked in.

    ``design_speed`` is in metres per second, one figure or one per point of an
    alignment whose speed varies along it. A road with no design speed has no
    limit -- there is always some speed at which any crest launches a car, and
    the answer to that is to say how fast the road is meant to be driven.
    """
    speed = np.asarray(design_speed, dtype='d')
    if weight_loss <= 0:
        return np.inf if speed.ndim else float('inf')
    with np.errstate(divide='ignore'):
        found = np.where(speed > 0, GRAVITY * weight_loss / (speed * speed),
                         np.inf)
    return found if speed.ndim else float(found)


def _limit_curvature(line: np.ndarray, maximum: Any,
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
    allowed = np.broadcast_to(np.asarray(maximum, dtype='d'), (count,))
    if count < 3 or not np.any(np.isfinite(allowed)):
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
        excess = np.abs(bend) - allowed * span
        shift = np.where(excess > 0.0,
                         np.sign(bend) * excess * CURVATURE_RELAXATION * lever,
                         0.0)
        if not closed:
            shift[0] = shift[-1] = 0.0
        if float(np.abs(shift).max()) < CURVATURE_TOLERANCE:
            break
        heights += shift
    return heights


def _settle_profile(line: np.ndarray, maximum_grade: Any,
                    maximum_curvature: Any, closed: bool) -> np.ndarray:
    """Bring an alignment inside the grade *and* the curvature limit.

    Rounding a crest off can steepen what leads to it, and clipping a grade puts
    a corner back into the profile, so neither limit can be applied once and
    left. Each is a projection onto a set of alignments -- both of them convex,
    both of them containing a level road -- so applying them in turn converges
    on one inside both. The curvature goes last, because a road half a percent
    too steep is still a road and a road with a kink in it is a ramp.
    """
    heights = line[:, 1]
    curvature = np.broadcast_to(np.asarray(maximum_curvature, dtype='d'),
                                (len(line),))
    if not np.any(np.isfinite(curvature)):
        return (_limit_grade(line, maximum_grade, closed=closed)
                if maximum_grade is not None else heights)
    # A closed course arrives with its first point repeated at the end, so the
    # two are one point with a step of nothing between them. Relaxing the bend
    # at a zero-length step is meaningless; the loop is settled over the points
    # it actually has, and the repeat takes the answer at the end.
    repeated = closed and bool(np.allclose(line[0, [0, 2]], line[-1, [0, 2]]))
    working = (line[:-1] if repeated else line).copy()
    grade = maximum_grade
    if repeated:
        # The repeat is the first point over again, and so is its limit.
        curvature = curvature[:-1]
        if grade is not None and np.asarray(grade).ndim:
            grade = np.asarray(grade, dtype='d')[:-1]
    for _round in range(SETTLING_ROUNDS):
        if grade is not None:
            working[:, 1] = _limit_grade(working, grade, closed=closed)
        working[:, 1] = _limit_curvature(working, curvature, closed=closed)
    if repeated:
        return np.concatenate([working[:, 1], working[:1, 1]])
    return working[:, 1]


def _smooth(values: np.ndarray, window: Any, closed: bool = False) -> np.ndarray:
    """A moving average over an alignment's heights, in points either side.

    ``window`` may be one width or **one per point**, which is what leaves the
    ground's own shape in one stretch of a road and irons it out of the next: a
    road smoothed to one width from end to end has one texture, and a lap of it
    feels the same everywhere.

    Edge-padded, so the first and last points of an open road stay on the ground
    the designer put them on; wrap-padded for a circuit, so the smoothing runs
    through the join instead of flattening towards it from both sides.

    A box filter of a width that changes per point is a difference of running
    sums rather than a convolution -- one pass over the alignment whatever the
    widths are, and exactly the average of the points each one covers.
    """
    widths = np.asarray(window, dtype='d').reshape(-1)
    reach = np.maximum((widths // 2).astype(np.intp), 0)
    if widths.size == 1:
        reach = np.full(len(values), int(reach[0]))
    if not len(reach) or int(reach.max()) < 1:
        return values
    padding = int(reach.max())
    padded = np.pad(values, padding, mode='wrap' if closed else 'edge')
    running = np.concatenate([[0.0], np.cumsum(padded)])
    at = np.arange(len(values)) + padding
    low, high = at - reach, at + reach + 1
    found: np.ndarray = (running[high] - running[low]) / (high - low)
    return found


def _limit_grade(line: np.ndarray, maximum: Any,
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
    # A step joins two points, and the limit over it is the gentler of what the
    # two allow: a stretch permitted to climb steeply does not drag the stretch
    # it joins up with it.
    allowed = np.broadcast_to(np.asarray(maximum, dtype='d'), (count,))
    over = np.minimum(allowed, np.roll(allowed, -1))
    for index in _forward(count, closed):
        behind = (index - 1) % count
        limit = over[behind] * steps[behind]
        heights[index] = np.clip(heights[index], heights[behind] - limit,
                                 heights[behind] + limit)
    for index in _backward(count, closed):
        ahead = (index + 1) % count
        limit = over[index] * steps[index]
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


def _ramp_up_to_grade(line: np.ndarray, maximum: Any,
                      closed: bool = False) -> np.ndarray:
    """Raise whatever is needed so no step exceeds the grade, lowering nothing.

    The companion to :func:`_limit_grade`, used after a floor has lifted part of
    an alignment: the approaches climb to meet the raised section instead of the
    section being dropped back down to them.
    """
    heights = line[:, 1].copy()
    steps = _steps(line, closed)
    count = len(heights)
    allowed = np.broadcast_to(np.asarray(maximum, dtype='d'), (count,))
    over = np.minimum(allowed, np.roll(allowed, -1))
    for index in _forward(count, closed):
        behind = (index - 1) % count
        heights[index] = max(heights[index],
                             heights[behind] - over[behind] * steps[behind])
    for index in _backward(count, closed):
        ahead = (index + 1) % count
        heights[index] = max(heights[index],
                             heights[ahead] - over[index] * steps[index])
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

    Where the road is carried *over* the land -- a deck, a causeway -- the ground
    is left exactly as it was found: a bridge stands over a valley, and filling
    the valley in would put the structure inside a hill of its own making.
    Where it is carried *inside* the land the opposite is true, and the cutting
    runs the length of the bore: see :meth:`RoadPath.reshaped_segments`. The
    road's ``ops`` say which stretches are which.

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
    half = path.widest() / 2.0
    reach = half + widening + maximum_earthwork

    laid = path.on_ground.all()
    reshaped = path.reshaped_segments()

    def conformed(x: Any, z: Any) -> np.ndarray:
        natural = np.asarray(height_fn(x, z), dtype='d')
        found = path.sample(x, z, radius=reach)
        distance = found.distance.reshape(natural.shape)
        road_height = found.height.reshape(natural.shape)
        near = distance < reach
        if not laid:
            near = near & reshaped[found.segment.reshape(natural.shape)]
        if not np.any(near):
            return natural
        # Beside a climbing road, the ground has to sit low enough that the
        # straight line a mesh draws between two samples cannot cross the
        # carriageway between them. The most the road can rise over one spacing
        # is its steepest grade, so that is how much further down the earthwork
        # goes at that resolution.
        outside = distance > half
        sag = np.where(outside, widening * path.max_grade, 0.0)
        section = (road_height
                   + path.section_offset(distance,
                                         found.side.reshape(natural.shape),
                                         found.bank.reshape(natural.shape))
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

    Where the road is on a bridge or a causeway or in a tunnel the layer writes
    the structure into the tile as well, and the carriageway over it takes the road's
    on-structure cut -- tapered onto it over ``transition``, so the verge
    flattens onto the deck across a few metres rather than stepping onto it.
    ``ground`` is the undisturbed terrain, which is where a bridge's piers stop;
    without it a deck is written with no piers under it.

    ``shade(x, z) -> sun`` is how much of the sun reaches each place, in [0, 1],
    written into the surface's vertex colours. A road through a wood carries the
    wood's shade rather than being a lit strip laid across it.

    ``structure_material`` is what a deck, a bore and a causeway -- its fill and
    the low wall on each edge of it alike -- are built from, and ``barrier`` the
    railing standing on the edge of a deck -- a
    different thing, and darker, because it is the object closest to the driver
    for the whole length of a crossing.
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
    #: How far along the centreline a lap begins, in metres. Zero for a road
    #: nobody races on, which is where its own start is anyway.
    start: float = 0.0
    #: What the road is posted at, in km/h; 0 for an unposted road. Written into
    #: the road's own extras rather than left to be guessed from the signs
    #: standing beside it, because it is a fact about the road and a game that
    #: reads the road should not have to read its furniture to learn it.
    posted: int = 0
    ground: HeightFn | None = None
    shade: HeightFn | None = None
    bridge: BridgeProfile | None = None
    causeway: CausewayProfile | None = None
    tunnel: TunnelProfile | None = None
    structure_material: PBRMaterial | None = None
    barrier: PBRMaterial | None = None
    transition: float = TRANSITION_LENGTH
    _material: PBRMaterial = field(init=False, repr=False)
    _surface: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.bridge = self.bridge or BridgeProfile()
        self.causeway = self.causeway or CausewayProfile()
        self.tunnel = self.tunnel or TunnelProfile()
        self.structure_material = (self.structure_material
                                   or concrete_material())
        self.barrier = self.barrier or barrier_material()
        if self.material is not None:
            self._material = self.material
            return
        # The surface is written beside the tileset and named by every tile,
        # so the material points at a file rather than carrying the pixels.
        self._surface = road_texture(self.texture_size, self.seed,
                                     self.path.profile)
        self._material = tarmac_material(
            wetness=self.wetness, profile=self.path.profile,
            image=ExternalImage(SURFACE_IMAGE, srgb=True))

    def metadata(self) -> dict[str, Any]:
        """The road itself, for the tileset's extras.

        A game cannot find a road in a pile of triangles, and needs it to put a
        car on the track, time a lap and drive an opponent round. So the road
        travels with the world it is baked into: its centreline, how wide it is,
        whether it closes into a circuit, its cross-section, and where along it
        the structures are -- which is how a game knows the car is on a bridge
        without asking the geometry.

        The cross-section is there because a game may have to build the
        carriageway itself: tile geometry is level-of-detail geometry and
        changes resolution as a car drives, and a collider whose surface steps
        under the wheels is a wall in the middle of an open road. See
        :class:`OpenGLContext.physics.road.RoadColliders`.

        The **lean** travels with the line, point for point, for the same
        reason: a game building the carriageway itself has to build the one the
        bake drew, and a collider swept flat under a superelevated corner is a
        surface the car falls through on the inside and stands on the outside.
        A road whose corners are flat writes an empty list rather than one zero
        per point.

        The line is written at ``metadata_spacing`` rather than at its full
        density -- a course is a shape, and a game re-samples it for whatever it
        is doing. Structures are given as distances along that same line.
        """
        line = self.path.resampled(self.metadata_spacing)
        closed = bool(np.allclose(self.path.points[0], self.path.points[-1]))
        walked = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        return {'roads': [{
            'name': self.name,
            'closed': closed,
            # Where a lap begins, in metres along the centreline. The grid, the
            # timing and the autopilot all count from here, so a game that
            # reads the road reads this with it.
            'start': round(float(self.start), 3),
            'carriagewayWidth': self.path.profile.carriageway_width,
            'totalWidth': self.path.profile.total_width,
            'posted': int(self.posted),
            'profile': _profile_json(self.path.profile),
            'length': self.path.length,
            'centreline': [[round(float(v), 3) for v in point] for point in line],
            'bank': ([round(float(lean), 5)
                      for lean in self.path.bank_at(walked)]
                     if self.path.banked else []),
            # How much more carriageway than `carriagewayWidth` the road has at
            # each of those points. Empty for a road of one width.
            'widening': ([round(float(extra), 3)
                          for extra in self.path.widening_at(walked)]
                         if self.path.wider else []),
            'structures': [{'kind': str(kind),
                            'from': round(start, 3),
                            'to': round(end, 3)}
                           for kind, start, end in self.path.structure_runs()],
        }], 'luminaires': self.luminaires()}

    def luminaires(self) -> list:
        """Where the lamps hang in this road's bores, as world points.

        The pool each one throws is baked onto the lining and costs nothing to
        draw, which is what lights a bore at any distance. What it cannot do is
        light anything *in* the bore -- a car driving through has no idea it is
        under a lamp -- so a game spends its few real lights on the fittings the
        driver is among, and needs to be told where those are. They travel in
        the tileset's extras for the same reason the road does: a lamp is not
        findable in a pile of triangles, and the tile it is in comes and goes.
        """
        found: list = []
        for kind, first, last in _op_runs(self.path.ops):
            if kind is not Op.TUNNEL or last - first < 1:
                continue
            run = self.path.points[first:last + 1]
            found.extend(
                [round(float(v), 3) for v in point]
                for point in tunnel_lamps(run, self.tunnel,
                                          bank=self.path.bank[first:last + 1]))
        return found

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

    def segments_in(self, region: BoundingBox, error: float) -> set[int]:
        """Which segments of the centreline this tile is responsible for.

        Indices into the line as re-sampled for a tile of this error, so
        segment *i* runs from point *i* to point *i+1*. A tileset refines by
        replacement, so the tiles at one level have to write each segment
        between them exactly once: a segment nobody writes is a carriageway
        that runs into the ground and stops the moment the viewer refines past
        the parent that had it, and one written twice is two coplanar surfaces
        fighting for the depth buffer.
        """
        line = self.path.resampled(self.spacing_for(error))
        found: set[int] = set()
        for rows in _runs_inside(line, region):
            found.update(range(int(rows[0]), int(rows[-1])))
        return found

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        reach = self.path.widest() * TILE_REACH
        if not self.path.crosses(region, margin=reach):
            return []
        spacing = self.spacing_for(error)
        line = self.path.resampled(spacing)
        stations = np.arange(len(line)) * (self.path.length / max(len(line) - 1, 1))
        ops = self.path.ops_at(stations)
        # The cut tapers onto a structure over `transition` metres, which is a
        # number of points at this tile's spacing: a coarse tile with a longer
        # spacing gets fewer of them and the same taper in the world.
        carried = np.array([op in CARRIED for op in ops], dtype='d')
        blend = _tapered(carried, max(int(round(self.transition / spacing)), 1))
        sections = morphed_sections(self.path.profile,
                                    self.path.profile.on_structure(), blend)
        # The lean at this tile's own spacing, and the camber it uses up. The
        # cut and the frame have to agree about the same point of the road, so
        # both are re-sampled from the alignment rather than one of them being
        # carried over from the density it was worked out at.
        bank = self.path.bank_at(stations)
        sections = widened_sections(sections,
                                    self.path.widening_at(stations),
                                    self.path.profile)
        sections = banked_sections(sections, bank, self.path.profile)

        found: list[SceneNode] = []
        for rows in _runs_inside(line, region):
            found.append(SceneNode(
                mesh=road_mesh(line[rows], self.path.profile,
                               material=self._material,
                               sections=sections[rows], bank=bank[rows],
                               shade=(None if self.shade is None
                                      else self._shade_of)),
                name=self.name))
            found.extend(self._structures(line, ops, rows, bank))
        return found

    def _shade_of(self, points: np.ndarray) -> Any:
        """How much of the sun reaches each written point of the surface."""
        assert self.shade is not None
        return np.asarray(self.shade(points[:, 0], points[:, 2]), dtype='d')

    def _structures(self, line: np.ndarray, ops: np.ndarray,
                    rows: np.ndarray, bank: np.ndarray) -> list[SceneNode]:
        """The decks and bores along the stretch of road a tile is drawing.

        A structure is split at the tile boundary the same way the carriageway
        is, so each tile carries the part of it that is inside the tile and no
        neighbour draws it twice.
        """
        out: list[SceneNode] = []
        inside = ops[rows]
        for kind, first, last in _op_runs(inside):
            if kind not in CARRIED or last - first < 1:
                continue
            run = line[rows[first:last + 1]]
            lean = bank[rows[first:last + 1]]
            if kind is Op.BRIDGE:
                parts = bridge_meshes(run, self.path.profile, self.ground,
                                      self.bridge, self.structure_material,
                                      self.barrier, bank=lean)
            elif kind is Op.CAUSEWAY:
                parts = causeway_meshes(run, self.path.profile, self.ground,
                                        self.causeway,
                                        self.structure_material, bank=lean)
            else:
                parts = tunnel_meshes(run, self.path.profile, self.tunnel,
                                      self.structure_material, bank=lean)
            out.extend(SceneNode(mesh=mesh, name='%s-%s' % (kind, part))
                       for part, mesh in parts.items())
        return out


def _cell_keys(x: np.ndarray, z: np.ndarray, cell: float) -> np.ndarray:
    """Which cell of a coarse grid each query falls in, as one integer."""
    return (np.floor(x / cell).astype(np.int64) << 32) \
        + np.floor(z / cell).astype(np.int64)


def _profile_json(profile: RoadProfile) -> dict[str, Any]:
    """A road's cross-section as a baked world carries it."""
    return {'laneWidth': profile.lane_width, 'lanes': profile.lanes,
            'shoulderWidth': profile.shoulder_width,
            'shoulderDrop': profile.shoulder_drop,
            'vergeWidth': profile.verge_width,
            'vergeDrop': profile.verge_drop,
            'crossfall': profile.crossfall,
            'textureLength': profile.texture_length}


def _op_runs(ops: np.ndarray) -> list[tuple[Op, int, int]]:
    """``(op, first, last)`` inclusive for each stretch of one operation."""
    out: list[tuple[Op, int, int]] = []
    start = 0
    for index in range(1, len(ops) + 1):
        if index < len(ops) and ops[index] is ops[start]:
            continue
        out.append((ops[start], start, index - 1))
        start = index
    return out


def _tapered(carried: np.ndarray, window: int) -> np.ndarray:
    """A 0/1 flag smoothed into a ramp over ``window`` points either side.

    What turns the change of cut at an abutment into a taper. The ramp reaches
    1 across the whole of the structure and 0 clear of it, so the carriageway
    is exactly the on-structure section over the deck and exactly the ordinary
    one away from it, with the change spread over the approach.
    """
    if window <= 1 or not len(carried):
        return carried
    kernel = np.ones(2 * window + 1) / float(2 * window + 1)
    padded = np.pad(carried, window, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='same')[window:-window]
    # Convolution rounds the corners off both ends of a run; taking the larger
    # of the two keeps the structure's own points at 1 and only ramps outside.
    ramp: np.ndarray = np.maximum(carried, np.clip(smoothed, 0.0, 1.0))
    return ramp


def _runs_inside(line: np.ndarray, region: BoundingBox) -> list[np.ndarray]:
    """Indices of the stretches of a centreline a tile is responsible for.

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
            out.append(np.arange(first, last))
    return out
