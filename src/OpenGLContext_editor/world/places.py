"""The places a road runs through, as regions a world's zones are made from.

A road knows what it is built as along its length -- its structures say where
the bores, the causeways and the bridges are -- and the trees beside it say
where it runs through forest. Each of those stretches is a different place to
be: lit differently, reflected differently, heard differently. This turns them
into :class:`Place` records, oriented boxes along the road, which
:mod:`OpenGLContext_editor.bake.zones` writes into a world as ``OGLC_zone``
nodes (see ``docs/zones.rst`` in OpenGLContext).

A stretch is cut into pieces no longer than its kind's ``chunk``, and each
piece's box is fitted to the centreline in it: along the chord from its first
point to its last, and wide enough across to hold the curve between. A bend
therefore takes a wider box rather than one that leaves the road.

A stretch of plain road is forest where there are enough trees beside it --
:data:`FOREST_TREES` within :data:`FOREST_REACH` metres, per hundred metres of
road. Elsewhere it is open ground and has no place: the world's own sky
lights it.

Everything here is arithmetic on the road and the tree table, with no GL and
no bake in it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from OpenGLContext_editor.world.structures import Op

__all__ = [
    'FOREST', 'TUNNEL', 'CAUSEWAY', 'BRIDGE', 'KINDS', 'Place', 'PlaceKind',
    'FOREST_REACH', 'FOREST_TREES', 'road_places',
]

FOREST = 'forest'
TUNNEL = 'tunnel'
CAUSEWAY = 'causeway'
BRIDGE = 'bridge'

#: How far from the centreline trees are counted, in metres.
FOREST_REACH = 45.0
#: How many trees within :data:`FOREST_REACH` a hundred metres of road needs to
#: be running through forest.
FOREST_TREES = 12.0

#: How far above the road the capture of a place's environment is taken, in
#: metres: where a driver's eyes and a car's roof are.
EYE_HEIGHT = 2.0


@dataclass(frozen=True)
class PlaceKind:
    """How large a place's box is, and how it meets its neighbours.

    ``chunk`` is the longest piece of road one box covers. ``across`` is how
    far the box reaches either side of the road, and ``above`` and ``below``
    how far above and below it, in metres. ``blend`` is how far outside the
    box the place fades out, and ``priority`` decides between overlapping
    places: a bore beats the forest over it.
    """

    chunk: float
    across: float
    above: float
    below: float
    blend: float
    priority: int


#: Each place's box. A bore is sized to the bore itself and fades over a few
#: metres at its portals; a causeway and a bridge reach out over the water or
#: the valley beside them; a forest reaches into the trees.
KINDS: dict[str, PlaceKind] = {
    TUNNEL: PlaceKind(chunk=80.0, across=2.0, above=1.5, below=1.0, blend=6.0,
                      priority=2),
    CAUSEWAY: PlaceKind(chunk=160.0, across=60.0, above=25.0, below=10.0,
                        blend=25.0, priority=1),
    BRIDGE: PlaceKind(chunk=160.0, across=50.0, above=25.0, below=40.0,
                      blend=25.0, priority=1),
    FOREST: PlaceKind(chunk=200.0, across=FOREST_REACH, above=30.0, below=6.0,
                      blend=30.0, priority=0),
}

#: The place each structure is.
_BY_OP = {Op.TUNNEL: TUNNEL, Op.CAUSEWAY: CAUSEWAY, Op.BRIDGE: BRIDGE}


@dataclass(frozen=True)
class Place:
    """One box of one place along the road.

    ``centre`` is the box's centre in the world. ``heading`` is its turn about
    +Y in radians, taking its local +X along the road; ``size`` is its full
    extent along the road, up and across it. ``eye`` is where on the road, in
    the box's own frame, the place is seen from -- where its environment is
    captured.
    """

    kind: str
    centre: tuple[float, float, float]
    heading: float
    size: tuple[float, float, float]
    eye: tuple[float, float, float]
    start: float
    end: float

    @property
    def rules(self) -> PlaceKind:
        return KINDS[self.kind]

    @property
    def rotation(self) -> tuple[float, float, float, float]:
        """The heading as a glTF quaternion ``(x, y, z, w)``."""
        half = self.heading / 2.0
        return (0.0, math.sin(half), 0.0, math.cos(half))


def road_places(road: Any, trees: Any | None = None,
                tunnel_half_width: float | None = None,
                tunnel_height: float = 7.5) -> list[Place]:
    """Every place along ``road``, from its structures and the trees beside it.

    ``road`` is a :class:`~OpenGLContext_editor.world.road.RoadPath`; its
    ``points``, ``stations`` and :meth:`structure_runs` are what is read.
    ``trees`` is the (N,3) table of trunk positions, or None for a world with
    none. ``tunnel_half_width`` and ``tunnel_height`` are the bore's, which
    size a tunnel's box; the half width defaults to the road's own.
    """
    points = np.asarray(road.points, dtype='d')
    stations = np.asarray(road.stations, dtype='d')
    if tunnel_half_width is None:
        tunnel_half_width = float(road.profile.total_width) / 2.0
    found: list[Place] = []
    covered = []
    for op, start, end in road.structure_runs():
        kind = _BY_OP.get(op)
        if kind is None or end - start <= 0.0:
            continue
        covered.append((start, end))
        found.extend(_pieces(kind, points, stations, start, end,
                             tunnel_half_width, tunnel_height))
    if trees is not None and len(trees):
        for start, end in _open_stretches(covered, float(stations[-1])):
            for piece in _split(start, end, KINDS[FOREST].chunk):
                if _wooded(trees, points, stations, *piece):
                    found.extend(_pieces(FOREST, points, stations, *piece,
                                         tunnel_half_width, tunnel_height))
    return found


def _open_stretches(covered: list, length: float) -> list:
    """The stretches of road no structure covers, in order."""
    out = []
    at = 0.0
    for start, end in sorted(covered):
        if start > at:
            out.append((at, start))
        at = max(at, end)
    if at < length:
        out.append((at, length))
    return out


def _split(start: float, end: float, chunk: float) -> list[tuple[float, float]]:
    count = max(1, int(math.ceil((end - start) / max(chunk, 1e-6))))
    edges = np.linspace(start, end, count + 1)
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


def _along(points: np.ndarray, stations: np.ndarray, start: float,
           end: float) -> np.ndarray:
    """The centreline from ``start`` to ``end``, its own points and both ends."""
    inside = points[(stations > start) & (stations < end)]
    first = np.array([np.interp(start, stations, points[:, i]) for i in range(3)])
    last = np.array([np.interp(end, stations, points[:, i]) for i in range(3)])
    return np.vstack([first, inside, last])


def _wooded(trees: Any, points: np.ndarray, stations: np.ndarray,
            start: float, end: float) -> bool:
    """Whether enough trees stand beside the road from ``start`` to ``end``."""
    line = _along(points, stations, start, end)
    table = np.asarray(trees, dtype='d')[:, [0, 2]]
    low = line[:, [0, 2]].min(axis=0) - FOREST_REACH
    high = line[:, [0, 2]].max(axis=0) + FOREST_REACH
    near = table[np.all((table >= low) & (table <= high), axis=1)]
    if not len(near):
        return False
    counted = np.zeros(len(near), dtype=bool)
    for a, b in zip(line[:-1, [0, 2]], line[1:, [0, 2]], strict=True):
        segment = b - a
        length2 = float(segment @ segment) or 1.0
        t = np.clip((near - a) @ segment / length2, 0.0, 1.0)
        gap = near - (a + t[:, None] * segment)
        counted |= np.einsum('ij,ij->i', gap, gap) <= FOREST_REACH ** 2
    return int(counted.sum()) >= FOREST_TREES * (end - start) / 100.0


def _pieces(kind: str, points: np.ndarray, stations: np.ndarray, start: float,
            end: float, tunnel_half_width: float,
            tunnel_height: float) -> list[Place]:
    rules = KINDS[kind]
    out = []
    for a, b in _split(start, end, rules.chunk):
        line = _along(points, stations, a, b)
        chord = line[-1, [0, 2]] - line[0, [0, 2]]
        length = float(np.hypot(*chord))
        if length < 1e-6:
            continue
        axis = chord / length
        side = np.array([-axis[1], axis[0]])
        flat = line[:, [0, 2]] - line[0, [0, 2]]
        along = flat @ axis
        across = flat @ side
        heights = line[:, 1]
        reach = tunnel_half_width + rules.across if kind == TUNNEL else rules.across
        above = tunnel_height + rules.above if kind == TUNNEL else rules.above
        low = np.array([along.min(), heights.min() - rules.below, across.min() - reach])
        high = np.array([along.max(), heights.max() + above, across.max() + reach])
        middle = (low + high) / 2.0
        size = high - low
        centre_xz = line[0, [0, 2]] + axis * middle[0] + side * middle[2]
        centre = (float(centre_xz[0]), float(middle[1]), float(centre_xz[1]))
        heading = math.atan2(-axis[1], axis[0])
        mid = _along(points, stations, (a + b) / 2.0, (a + b) / 2.0)[0]
        eye_flat = mid[[0, 2]] - centre_xz
        eye = (float(eye_flat @ axis), float(mid[1] + EYE_HEIGHT - middle[1]),
               float(eye_flat @ side))
        out.append(Place(kind, centre, heading,
                         (float(size[0]), float(size[1]), float(size[2])),
                         eye, a, b))
    return out
