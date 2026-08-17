"""What gets built where a road leaves the ground.

An alignment settled for grade and for the speed it is meant to be driven at
does not lie on the land: it runs above it across the low ground and below it
through the high ground, by whatever the smoothing and the grade limit demand.
Small departures are absorbed by the ground itself -- a cutting on one side, an
embankment on the other, both produced by
:func:`~OpenGLContext_editor.world.road.conform_terrain`. Large ones cannot be:
an embankment ninety metres high is a hill of imported soil standing where a
valley was, and a cutting ninety metres deep is a mountain sliced open.

So a departure past a threshold becomes a *structure* -- a bridge over the low
ground or a tunnel through the high ground -- and the terrain under or over it
is left as it is. This module makes that choice. The geometry it leads to is
:mod:`OpenGLContext.scenegraph.roadworks`, and the ground's side of the bargain
is ``conform_terrain``'s ``structures`` argument.

The choice is deliberately made on the finished alignment rather than while it
is being settled: what a bridge or a tunnel costs does not change where the road
wants to go, and settling first means the same alignment can be re-examined
under different thresholds without being recomputed.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    'Op', 'Structure', 'choose_structures',
    'CUTTING_LIMIT', 'EMBANKMENT_LIMIT', 'MINIMUM_TUNNEL', 'MINIMUM_SPAN',
    'APPROACH_LIMIT', 'ABUTMENT_HEIGHT', 'PORTAL_COVER', 'MINIMUM_CAUSEWAY',
    'MINIMUM_APPROACH',
]

#: How deep a cutting is worth digging, in metres. Below this the road is bored
#: through instead: the spoil from a deeper cut has to go somewhere, the faces
#: have to be held, and the landscape the road was routed through stops being
#: there. Motorway practice puts the crossover in the high teens.
CUTTING_LIMIT = 18.0

#: How tall an embankment is worth building, in metres. Above this the fill --
#: which grows as the square of the height, because it batters out on both
#: sides -- costs more than a deck on piers, and it dams whatever the valley
#: was draining.
EMBANKMENT_LIMIT = 14.0

#: How long a departure has to last before a structure is worth its portals or
#: its abutments, in metres. A short deep dip is dug out; a short high hop is
#: filled.
MINIMUM_TUNNEL = 60.0
MINIMUM_SPAN = 40.0

#: How far a structure reaches out past the threshold that chose it, in metres,
#: to land on its abutments or open at its portals. A deck stops where it meets
#: the ground rather than in mid-air, so the run is extended down the approach
#: until the departure falls to ``ABUTMENT_HEIGHT`` (or the cover falls to
#: ``PORTAL_COVER``) -- and no further than ``APPROACH_LIMIT``, so a departure
#: that decays over a kilometre does not turn the whole road into bridge.
APPROACH_LIMIT = 120.0
ABUTMENT_HEIGHT = 2.0
PORTAL_COVER = 2.0

#: How long a road has to run over drowned ground before the fill carrying it
#: is called a causeway rather than an embankment, in metres.
MINIMUM_CAUSEWAY = 40.0

#: The shortest stretch of ordinary road worth building between two structures,
#: in metres. Less than this is an island of made ground inside a hillside that
#: was otherwise left alone -- which is a wall across the road, and is not how
#: anything is built: a deck lands on the portal it runs into.
MINIMUM_APPROACH = 60.0


class Op(Enum):
    """What is built along one stretch of a road.

    ``DIRT`` is the ordinary case and the default everywhere else: the road
    follows the ground and the ground is reshaped to meet it. The other three
    are structures, and each of them tells the terrain to leave that stretch
    alone.
    """

    DIRT = 'dirt'
    CAUSEWAY = 'causeway'
    BRIDGE = 'bridge'
    TUNNEL = 'tunnel'

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Structure:
    """One stretch of a road, and what is built along it.

    ``first`` and ``last`` are inclusive indices into the alignment. On a
    circuit a structure may straddle the point the array happens to begin at,
    in which case ``last`` is *before* ``first`` and :attr:`wraps` is true; use
    :meth:`indices` rather than slicing.
    """

    kind: Op
    first: int
    last: int
    wraps: bool = False

    def __repr__(self) -> str:
        return ('Structure(%s, %d..%d%s)'
                % (self.kind.value, self.first, self.last,
                   ', wrapping' if self.wraps else ''))

    def indices(self, count: int) -> np.ndarray:
        """Every alignment index this stretch covers, in order along the road."""
        if self.wraps:
            return np.concatenate([np.arange(self.first, count),
                                   np.arange(0, self.last + 1)])
        return np.arange(self.first, self.last + 1)

    def length(self, line: Any) -> float:
        """How long the stretch is, following the alignment."""
        points = np.asarray(line, dtype='d')[self.indices(len(line))]
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def choose_structures(line: Any, natural: Any, *,
                      cutting_limit: float = CUTTING_LIMIT,
                      embankment_limit: float = EMBANKMENT_LIMIT,
                      minimum_tunnel: float = MINIMUM_TUNNEL,
                      minimum_span: float = MINIMUM_SPAN,
                      approach_limit: float = APPROACH_LIMIT,
                      waterline: float | None = None,
                      closed: bool = False,
                      overrides: Iterable[tuple[int, int, Op]] = (),
                      ) -> list[Structure]:
    """Partition an alignment into the operations that build it.

    ``line`` is the settled alignment as (N,3) world points and ``natural`` the
    height of the *undisturbed* ground beneath each of them -- the terrain
    before the road did anything to it. The result covers every point exactly
    once, in order along the road, so ``[s.kind for s in result]`` is the
    sequence of operations and nothing between two structures is unaccounted
    for.

    ``waterline`` marks fill over drowned ground as :attr:`Op.CAUSEWAY` rather
    than plain dirt; without one, no causeway is found, because a road on fill
    over dry ground is an embankment.

    ``closed`` says the alignment is a circuit, so a structure straddling the
    point the array starts at is one structure rather than two.

    ``overrides`` are ``(first, last, op)`` triples a designer has decided,
    applied after the ground has had its say and inclusive at both ends.
    """
    points = np.asarray(line, dtype='d').reshape(-1, 3)
    ground = np.asarray(natural, dtype='d').reshape(-1)
    if len(points) != len(ground):
        raise ValueError("the alignment and the ground under it differ in length")
    if len(points) < 2:
        raise ValueError("an alignment needs at least two points")

    departure = points[:, 1] - ground
    station = _stations(points)
    kinds = np.full(len(points), Op.DIRT, dtype=object)
    kinds[departure < -cutting_limit] = Op.TUNNEL
    kinds[departure > embankment_limit] = Op.BRIDGE
    _drop_short(kinds, station, Op.TUNNEL, minimum_tunnel, closed)
    _drop_short(kinds, station, Op.BRIDGE, minimum_span, closed)
    _reach_out(kinds, station, departure, Op.BRIDGE, approach_limit,
               departure > ABUTMENT_HEIGHT, closed)
    _reach_out(kinds, station, departure, Op.TUNNEL, approach_limit,
               departure < -PORTAL_COVER, closed)
    _close_gaps(kinds, station, closed)
    if waterline is not None:
        _mark_causeways(kinds, station, ground, departure, waterline, closed)
    for first, last, op in overrides:
        kinds[first:last + 1] = op
    return _runs(kinds, closed)


def _stations(points: np.ndarray) -> np.ndarray:
    """Distance along the alignment to each point."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _spans(kinds: np.ndarray, wanted: Op, closed: bool
           ) -> list[tuple[int, int, bool]]:
    """``(first, last, wraps)`` for every run of ``wanted`` in ``kinds``."""
    found = _runs_of(kinds == wanted)
    if closed and len(found) > 1 and found[0][0] == 0 \
            and found[-1][1] == len(kinds) - 1:
        first, last = found[-1][0], found[0][1]
        return [(first, last, True)] + [(a, b, False) for a, b in found[1:-1]]
    return [(a, b, False) for a, b in found]


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` inclusive for every run of ``True``."""
    edges = np.flatnonzero(np.diff(np.concatenate(
        [[False], mask.astype(bool), [False]]).astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2],
                                                  strict=True)]


def _covered(first: int, last: int, wraps: bool, count: int) -> np.ndarray:
    if wraps:
        return np.concatenate([np.arange(first, count), np.arange(0, last + 1)])
    return np.arange(first, last + 1)


def _run_length(station: np.ndarray, first: int, last: int, wraps: bool) -> float:
    """How long a run is along the road, measured between its end points."""
    if wraps:
        return float(station[-1] - station[first] + station[last])
    return float(station[last] - station[first])


def _drop_short(kinds: np.ndarray, station: np.ndarray, wanted: Op,
                minimum: float, closed: bool) -> None:
    """Turn runs too short to be worth their portals back into dirt."""
    for first, last, wraps in _spans(kinds, wanted, closed):
        if _run_length(station, first, last, wraps) < minimum:
            kinds[_covered(first, last, wraps, len(kinds))] = Op.DIRT


def _reach_out(kinds: np.ndarray, station: np.ndarray, departure: np.ndarray,
               wanted: Op, limit: float, eligible: np.ndarray,
               closed: bool) -> None:
    """Extend each run of ``wanted`` down its approaches to land on the ground.

    A deck stops on an abutment and a bore opens at a portal, both of which sit
    where the road is nearly back on the land -- not at the threshold that chose
    the structure, which is part way up the approach. So each end walks outwards
    while the road is still ``eligible`` (clear of the ground by more than an
    abutment's height, or under it by more than a portal's cover), for at most
    ``limit`` metres, and stops short of any point already spoken for.
    """
    count = len(kinds)
    for first, last, _wraps in _spans(kinds, wanted, closed):
        for step, end in ((-1, first), (+1, last)):
            at = end
            while True:
                nxt = at + step
                if closed:
                    nxt %= count
                elif not 0 <= nxt < count:
                    break
                if kinds[nxt] is not Op.DIRT or not eligible[nxt]:
                    break
                if abs(_run_length(station, min(end, nxt), max(end, nxt), False)) \
                        > limit:
                    break
                kinds[nxt] = wanted
                at = nxt
                if at == end:                    # pragma: no cover - ring closed
                    break


def _close_gaps(kinds: np.ndarray, station: np.ndarray, closed: bool) -> None:
    """Give a short stretch of road between two structures to the first of them.

    A deck that stops ten metres short of a portal leaves ten metres of
    embankment standing inside a hillside nothing else disturbed, which is a
    wall across the road and is not how anything is built. The stretch goes to
    the structure before it, so the deck lands on the portal.
    """
    count = len(kinds)
    for first, last in _runs_of(kinds == Op.DIRT):
        if _run_length(station, first, last, False) >= MINIMUM_APPROACH:
            continue
        before = kinds[first - 1] if first > 0 else (
            kinds[-1] if closed else Op.DIRT)
        after = kinds[last + 1] if last + 1 < count else (
            kinds[0] if closed else Op.DIRT)
        if before is Op.DIRT or after is Op.DIRT:
            continue
        kinds[first:last + 1] = before


def _mark_causeways(kinds: np.ndarray, station: np.ndarray, ground: np.ndarray,
                    departure: np.ndarray, waterline: float,
                    closed: bool) -> None:
    """Label fill carried over drowned ground as a causeway.

    Not every embankment is one: what makes a causeway is that the land it
    stands on is under water, so the run has to be long enough to be a crossing
    rather than a puddle, and the road has to be on fill rather than in cut.
    """
    over_water = (ground < waterline) & (departure > 0.0) & (kinds == Op.DIRT)
    for first, last in _runs_of(over_water):
        if _run_length(station, first, last, False) >= MINIMUM_CAUSEWAY:
            kinds[first:last + 1] = Op.CAUSEWAY
    if closed and len(kinds) > 1 and kinds[0] is Op.CAUSEWAY:
        pass                                     # the wrap is handled by _runs


def _runs(kinds: np.ndarray, closed: bool) -> list[Structure]:
    """Adjacent points of one kind gathered into structures, in road order."""
    changes = [0] + [i for i in range(1, len(kinds)) if kinds[i] is not kinds[i - 1]]
    spans = [(changes[i], (changes[i + 1] - 1) if i + 1 < len(changes)
              else len(kinds) - 1) for i in range(len(changes))]
    if closed and len(spans) > 1 and kinds[0] is kinds[-1]:
        first = spans[-1][0]
        last = spans[0][1]
        wrapped = Structure(kinds[0], first, last, wraps=True)
        return [wrapped] + [Structure(kinds[a], a, b) for a, b in spans[1:-1]]
    return [Structure(kinds[a], a, b) for a, b in spans]


def structure_mask(structures: Sequence[Structure], count: int,
                   kinds: Iterable[Op] = (Op.BRIDGE, Op.TUNNEL)) -> np.ndarray:
    """Which alignment points are carried by one of ``kinds``.

    What ``conform_terrain`` reads: the ground is left alone where the road is
    on a structure, because a bridge stands over the land and a tunnel runs
    inside it.
    """
    wanted = set(kinds)
    mask = np.zeros(count, dtype=bool)
    for structure in structures:
        if structure.kind in wanted:
            mask[structure.indices(count)] = True
    return mask
