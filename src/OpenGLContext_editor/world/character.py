"""What kind of road this is, here.

A road is laid out to a design speed, a grade limit, a smoothing window and a
cleared width, and if each of those is one figure from end to end then the road
has one character: a lap of it is the same lap however long it is, and a driver
who has taken the first corner has taken them all.

Real roads are not like that, and the reason is not decoration. A road's
character changes because *what it is crossing* changes and because *what a
driver needs* changes with it -- a hillside forces a climb, a tight corner is
not worth ironing flat for a speed nothing takes it at, a bend nobody can see
round has to be opened out before anybody can pass on it. So everything here is
**derived**: from the corner the road is on, from the land under it, and from
how far ahead a driver has to be able to see.

:func:`corner_radii`
    a radius per corner of a plan, so a lap has a hairpin on it and a sweeper
    and a good many corners of the road's own kind
:func:`road_character`
    the four limits that vary along an alignment, worked out from that plan and
    the ground it crosses, ready to hand to
    :func:`~OpenGLContext_editor.world.road.follow_terrain`

Nothing here draws a road or moves one. It decides what the road is *for*, in
each of its stretches, and the alignment machinery does the rest.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.road import (
    CAUTION,
    corner_speed,
    plan_curvature,
    resample_polyline,
)

from OpenGLContext_editor.world.road import points_along
from OpenGLContext_editor.world.signs import stopping_distance

HeightFn = Callable[[Any, Any], Any]

__all__ = ['RoadCharacter', 'corner_radii', 'road_character', 'CORNER_MIX',
           'TIGHTEST_CORNER', 'CLIMB_REACH', 'MOST_CLEARING',
           'LEAST_CLEARING', 'SIGHT_FROM', 'SMOOTHING_AT_SPEED',
           'LEAST_SMOOTHING', 'CLIMBING_LANE_GRADE', 'CLIMBING_LANE_RUN',
           'CLIMBING_LANE_TAPER', 'CLIMBING_LANE_SHARE', 'CLIMB_SMOOTHING']

#: What a lap's corners are made of: how tight each kind is against the road's
#: own design corner, and how many of them there are for every ten corners.
#:
#: The mix is what a lap is. All at 1.0 and the road is one corner repeated; a
#: lap of nothing but hairpins is a car park. What is here is a road that is
#: *mostly* what it was laid out to be, with a corner or two that has to be
#: braked for and a corner or two that can be carried flat -- which is what
#: gives a lap somewhere to gain time and somewhere to lose it.
CORNER_MIX: tuple[tuple[float, float], ...] = (
    (0.22, 1.0),                                 # a hairpin
    (0.50, 2.0),                                 # slow: third gear and patience
    (1.00, 4.0),                                 # the road's own corner
    (1.90, 2.0),                                 # quick: lift, do not brake
    (3.20, 1.0),                                 # a sweeper taken flat
)

#: The tightest corner worth building, in metres. Below this a road is a track
#: through a car park: the arc is shorter than the car, and a driver spends
#: longer turning the wheel than driving round it.
TIGHTEST_CORNER = 45.0

#: Over how much road the land's own climb is measured when deciding whether a
#: stretch is a hillside, in metres. A road is routed at the scale of the
#: hillside rather than the hummocks on it, and a climb shorter than this is a
#: bump in the ground rather than a hill to be got over.
CLIMB_REACH = 240.0

#: How far the trees are cut back **from the centreline** at most and at least,
#: in metres. The floor is the corridor a two-lane road is built inside -- its
#: own width and the strip beside it -- and a caller with a road of its own
#: passes that road's. The ceiling is what a *forest* road is: past it the drive
#: is past the trees rather than through them, however much sight it would buy.
MOST_CLEARING = 18.0
LEAST_CLEARING = 7.0

#: How far inside the centreline a driver's eye is, in metres -- the middle of
#: the lane they are in. Sight round a bend is measured from *there* to the
#: obstruction, so the corridor that buys it is that much wider than the offset
#: the sight line needs.
SIGHT_FROM = 1.8

#: How many metres of alignment are averaged into the height of each point at
#: the road's full design speed, and the least any stretch gets however slow it
#: is. A bump taken at two hundred is a car in the air, so a fast road is ironed
#: flat; the same bump at eighty is a road with some shape in it, and ironing it
#: out there costs the drive and buys nothing.
SMOOTHING_AT_SPEED = 60.0
LEAST_SMOOTHING = 8.0

#: What earns a stretch an extra lane's worth of carriageway: how steeply it
#: climbs and for how far, and over how many metres the road opens out to it and
#: closes again.
#:
#: A climbing lane is a real road's answer to a real problem -- on a long climb
#: the slow traffic is much slower than the quick, and without somewhere to pass
#: everything arrives at the top in one queue. One in twenty-five for four
#: hundred metres is about where a road earns one.
CLIMBING_LANE_GRADE = 0.04
CLIMBING_LANE_RUN = 400.0
CLIMBING_LANE_TAPER = 90.0

#: The most of a road that is built wide enough to be passed on, as a share of
#: its length. Hilly country asks for more climbing lanes than anybody builds:
#: what decides how many there are is what a road authority will pay for, and
#: the answer is the worst of the climbs rather than all of them. A road widened
#: along half its length is a wide road rather than a road with passing places
#: on it.
CLIMBING_LANE_SHARE = 0.15

#: Over how much road the land's grade is read when looking for a climb, in
#: metres. Long enough that a hummock does not break a climb in two or make one
#: out of nothing; short enough that the ends of a climb are where they are.
CLIMB_SMOOTHING = 90.0

#: The slowest a stretch of road is ever laid out for, in metres per second. A
#: design speed of nothing asks for an infinitely sharp crest, and there is no
#: corner so tight that a road through it stops being a road.
SLOWEST_DESIGN = 40.0 / 3.6


@dataclass(frozen=True)
class RoadCharacter:
    """How a road's design varies along its own length, point by point.

    One entry per point of the alignment
    :func:`~OpenGLContext_editor.world.road.follow_terrain` will produce at the
    same spacing -- :func:`~OpenGLContext_editor.world.road.points_along` says
    how many that is -- so the four go straight in as its ``design_speed``,
    ``maximum_grade`` and ``smoothing``, and the fourth travels with the
    finished road as how far the trees are cut back beside it.
    """

    #: How fast each stretch is laid out for, in metres per second. What rounds
    #: its crests off, and never more than the road as a whole is for.
    design_speed: np.ndarray
    #: How steeply each stretch may climb, as a fraction.
    grade_limit: np.ndarray
    #: How many metres of alignment are averaged into each point's height.
    smoothing: np.ndarray
    #: How far the trees are cut back from the centreline, in metres.
    clearance: np.ndarray
    #: How much more carriageway each stretch has, in metres -- zero for the
    #: road's own width, a lane's worth where it is built to be passed on.
    widening: np.ndarray

    def varies(self) -> bool:
        """Whether this is a road of more than one character.

        A road that came out uniform is worth knowing about: it is the road the
        limits alone would have given, and nothing downstream has to carry an
        array of one repeated figure to build it.
        """
        return any(float(np.ptp(each)) > 1e-9
                   for each in (self.design_speed, self.grade_limit,
                                self.smoothing, self.clearance,
                                self.widening))

    def summary(self) -> dict[str, dict[str, float]]:
        """The range each of the four covers, for a report or a bake's log.

        What a designer wants to know about a generated road is not the arrays
        but whether it got a hairpin, a climb and a clearing -- which is the
        spread of these four, in the units they are in.
        """
        return {name: {'least': float(np.min(values)),
                       'most': float(np.max(values)),
                       'mean': float(np.mean(values))}
                for name, values in (('designSpeed', self.design_speed),
                                     ('gradeLimit', self.grade_limit),
                                     ('smoothing', self.smoothing),
                                     ('clearance', self.clearance),
                                     ('widening', self.widening))}


def corner_radii(plan: Any, design_radius: float,
                 mix: Any = CORNER_MIX, spread: float = 1.0,
                 tightest: float = TIGHTEST_CORNER,
                 seed: int = 0) -> np.ndarray:
    """A radius per vertex of a plan: what kind of corner each one is to be.

    ``design_radius`` is the corner the road is laid out to -- what
    :func:`~OpenGLContext.scenegraph.road.cornering_radius` asks for at the
    design speed. Each vertex is drawn from ``mix``, a sequence of
    ``(multiple of the design radius, how many per ten corners)``, so most
    corners are the road's own and a few are not. ``spread`` scales how far the
    drawn corners depart from the design radius: 0 gives every corner the design
    radius, which is the road one figure would have given.

    Nothing tighter than ``tightest`` comes out, whatever the mix asks for: a
    corner shorter than the car going round it is not a corner.

    Hand the result to
    :func:`~OpenGLContext_editor.world.route.hold_corners`, which rounds each
    vertex to the radius it drew. The **turn angle** stays the plan's own, so
    what a designer or a generator drew is still where the road goes; what
    changes is how hard each of those turns has to be taken.
    """
    plan = np.asarray(plan, dtype='d').reshape(-1, 2)
    multiples = np.asarray([multiple for multiple, _weight in mix], dtype='d')
    weights = np.asarray([weight for _multiple, weight in mix], dtype='d')
    if not len(multiples) or weights.sum() <= 0:
        raise ValueError("a corner mix needs at least one kind with a weight")
    reach = float(np.clip(spread, 0.0, 1.0))
    # Drawn in order and then shuffled, rather than drawn independently: over
    # the eight or ten corners a lap has, independent draws leave whole laps
    # with no hairpin on them at all, which is the one thing the mix is for.
    generator = np.random.default_rng(seed)
    share = weights / weights.sum() * len(plan)
    counts = _whole_shares(share, generator)
    drawn = np.repeat(multiples, counts)
    generator.shuffle(drawn)
    found: np.ndarray = design_radius * (1.0 + reach * (drawn - 1.0))
    return np.maximum(found, float(tightest))


def _whole_shares(share: np.ndarray, generator: Any) -> np.ndarray:
    """Turn fractional shares of a lap's corners into whole numbers of them.

    The whole part of each share, and then the remaining corners handed out to
    the kinds with the most left over. A lap of nine corners cannot give four
    tenths of one to the hairpins, and rounding each share on its own either
    loses a corner or invents one.
    """
    whole = np.floor(share).astype(int)
    short = int(round(float(share.sum()))) - int(whole.sum())
    if short > 0:
        order = np.argsort(-(share - whole))
        whole[order[:short]] += 1
    elif short < 0:                              # pragma: no cover - rounding
        order = np.argsort(share - whole)
        whole[order[:-short]] -= 1
    return whole


def road_character(plan: Any, height_fn: HeightFn, design_speed: float,
                   spacing: float = 5.0, closed: bool = False,
                   grade_limit: float = 0.10,
                   steep_grade: float = 0.14,
                   smoothing: float = SMOOTHING_AT_SPEED,
                   least_smoothing: float = LEAST_SMOOTHING,
                   clearance: float = LEAST_CLEARING,
                   most_clearing: float = MOST_CLEARING,
                   sight_from: float = SIGHT_FROM,
                   caution: float = CAUTION,
                   climbing_lane: float = 0.0,
                   lane_grade: float = CLIMBING_LANE_GRADE,
                   lane_run: float = CLIMBING_LANE_RUN,
                   lane_taper: float = CLIMBING_LANE_TAPER,
                   lane_share: float = CLIMBING_LANE_SHARE) -> RoadCharacter:
    """What kind of road each stretch of this plan is to be.

    ``plan`` is the alignment in plan -- (N,2) as XZ, or (N,3) whose height is
    ignored -- already rounded to the corners it is going to have
    (:func:`~OpenGLContext_editor.world.route.hold_corners`), because what
    varies is worked out from those corners. ``height_fn`` is the land it
    crosses, and ``spacing`` and ``closed`` must be the ones
    :func:`~OpenGLContext_editor.world.road.follow_terrain` is about to be given,
    or the arrays will not line up with the alignment they are for.

    Four things come out, and each is derived:

    **How fast a stretch is for** is what its own corner allows
    (:func:`~OpenGLContext.scenegraph.road.corner_speed`), never more than
    ``design_speed`` and never less than a road is worth building. It is what
    rounds the crests off, and taking it from the corner is what stops a hairpin
    being handed the vertical curve of a two-hundred-an-hour straight -- a
    quarter of a kilometre of earthwork for a crest nobody meets at that speed.

    **How steeply it may climb** is ``grade_limit``, raised towards
    ``steep_grade`` where the land itself climbs harder than that over
    :data:`CLIMB_REACH`. A road held to a gentle grade across a hillside stands
    off it on an embankment for as far as the hillside lasts; allowed the
    hillside's own grade it climbs with it, which is a climb to drive rather
    than a viaduct to sit on.

    **How much it is smoothed** follows the speed: ``smoothing`` metres of
    averaging where the road is for its full design speed, falling towards
    ``least_smoothing`` where it is slow. A bump taken at two hundred is a car
    in the air and has to go; the same bump at eighty is the road having some
    shape, and ironing it out costs the drive and buys nothing.

    **How far the trees are cut back**, from the centreline, is what a driver
    needs to see round the bend they are on. Sight round a bend of radius *r*
    past an obstruction ``clear`` to the side is about ``sqrt(8 * r * clear)``,
    so the offset that buys a stopping distance is ``distance**2 / (8 * r)`` --
    and the *corridor* that buys it is that much further out again than
    ``sight_from``, which is how far inside the centreline the driver is sitting.
    Worked out for the speed a *careful* driver takes the bend at (``caution``),
    floored at ``clearance`` -- the corridor the road is built inside anyway --
    and capped at ``most_clearing``, past which the drive is past the trees
    rather than through them.

    Where that lands is worth knowing: it is the corners **near the design
    radius** that get opened out. A tighter one is taken slowly enough to see
    round already, and a much wider one is straight enough. So a lap gets its
    clearings at the corners that are quick but not flat -- which are the ones a
    driver most needs to see the exit of.

    **How much wider it is** is ``climbing_lane`` metres of extra carriageway on
    the sustained climbs -- ``lane_grade`` or more for ``lane_run`` metres --
    tapered in and out over ``lane_taper``. It is a real road's answer to a real
    problem: on a long climb what is slow is much slower than what is quick, and
    a road with nowhere to pass delivers the whole of its traffic to the top in
    one queue. Zero, the default, is a road of one width.

    Not *every* climb: hill country asks for more climbing lanes than anybody
    builds, and a road widened along half its length is a wide road rather than
    a road with passing places on it. ``lane_share`` is how much of the road may
    be widened, and the climbs that get it are the worst ones -- longest and
    steepest first -- until that is spent.
    """
    line = _resampled(plan, spacing, closed)
    curvature = np.abs(plan_curvature(line, closed=closed))
    with np.errstate(divide='ignore'):
        radius = 1.0 / np.maximum(curvature, 1e-12)
    allowed = np.array([corner_speed(float(each)) for each in radius])
    speed = np.clip(allowed, SLOWEST_DESIGN, float(design_speed))
    # How much of the road's full speed each stretch is worth, which is what
    # both the smoothing and the clearing are scaled by.
    share = speed / max(float(design_speed), 1e-9)
    return RoadCharacter(
        design_speed=speed,
        grade_limit=_climbing(line, height_fn, grade_limit, steep_grade),
        smoothing=(least_smoothing
                   + (float(smoothing) - least_smoothing) * share ** 2),
        clearance=_sight_clearing(radius, speed, caution, clearance,
                                  most_clearing, sight_from),
        widening=_climbing_lane(line, height_fn, climbing_lane, lane_grade,
                                lane_run, lane_taper, lane_share, closed))


def _resampled(plan: Any, spacing: float, closed: bool) -> np.ndarray:
    """The plan at the spacing the alignment will be built at, as (N,3).

    The same re-sampling :func:`~OpenGLContext_editor.world.road.follow_terrain`
    does, so what comes out is one figure per point of the alignment these
    limits are for and the two cannot fall out of step.
    """
    found = np.asarray(plan, dtype='d')
    if found.shape[1] == 2:
        found = np.stack([found[:, 0], np.zeros(len(found)), found[:, 1]],
                         axis=-1)
    if closed and not np.allclose(found[0], found[-1]):
        found = np.vstack([found, found[:1]])
    line: np.ndarray = resample_polyline(found, spacing)
    assert len(line) == points_along(plan, spacing, closed)
    return line


def _climbing(line: np.ndarray, height_fn: HeightFn, ordinary: float,
              steep: float, reach: float = CLIMB_REACH) -> np.ndarray:
    """How steeply each point of an alignment may climb, as a fraction.

    The land's own grade over ``reach`` metres of road, which is the scale a
    hillside is read at: a rise and a fall inside that is a hummock the road
    goes over, and a rise that lasts the whole of it is a hill the road has to
    get up. Where the land is gentler than ``ordinary`` nothing is raised, so
    flat country gets the road's own grade and a mountainside gets as much of
    ``steep`` as it needs and no more.
    """
    ground = np.asarray(height_fn(line[:, 0], line[:, 2]), dtype='d')
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    spacing = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1.0
    apart = max(int(round(float(reach) / max(spacing, 1e-9) / 2.0)), 1)
    behind = np.clip(np.arange(len(ground)) - apart, 0, len(ground) - 1)
    ahead = np.clip(np.arange(len(ground)) + apart, 0, len(ground) - 1)
    run = np.maximum(walked[ahead] - walked[behind], 1e-9)
    demanded = np.abs(ground[ahead] - ground[behind]) / run
    return np.clip(demanded, float(ordinary), float(steep))


def _climbing_lane(line: np.ndarray, height_fn: HeightFn, extra: float,
                   grade: float, run: float, taper: float, share: float,
                   closed: bool) -> np.ndarray:
    """How much wider the carriageway is at each point, in metres.

    ``extra`` on the climbs worth widening for, nothing anywhere else, with the
    change spread over ``taper`` metres at each end -- a road that stepped from
    two lanes to three at a vertex would have a lane beginning in mid-air.

    A climb is a **contiguous stretch** of at least ``run`` metres over which
    the land rises at ``grade`` or more, read at the scale of a hillside rather
    than the hummocks on it (:data:`CLIMB_SMOOTHING`). They are then taken
    worst-first -- by how much height each makes a slow vehicle gain, which is
    what costs it its speed -- until ``share`` of the road has been widened.

    The climb is read off the *land*, not off the alignment, for the same reason
    the grade limit is: the alignment is what comes out of these limits, and
    asking it about itself before it exists is asking the wrong question.
    """
    if extra <= 0.0 or len(line) < 3:
        return np.zeros(len(line))
    ground = np.asarray(height_fn(line[:, 0], line[:, 2]), dtype='d')
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    spacing = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1.0
    window = max(int(round(CLIMB_SMOOTHING / max(spacing, 1e-9))), 1)
    rising = _smoothed(np.gradient(ground, np.maximum(walked, 0.0)), window,
                       closed)
    # Climbing rather than falling: what is slow going up a hill is not slow
    # coming down it.
    climbs = [(first, last) for first, last in _runs_of(rising >= float(grade),
                                                        closed)
              if walked[last] - walked[first] >= float(run)]
    if not climbs:
        return np.zeros(len(line))
    # Worst first: how much height a slow vehicle has to gain on each, which is
    # what decides how much speed it loses and so how long the queue behind it.
    climbs.sort(key=lambda span: ground[span[1]] - ground[span[0]], reverse=True)
    budget = float(share) * float(walked[-1])
    earned = np.zeros(len(line), dtype=bool)
    spent = 0.0
    for first, last in climbs:
        length = walked[last] - walked[first]
        if spent + length > budget and spent > 0.0:
            continue
        earned[first:last + 1] = True
        spent += length
    return float(extra) * _tapered(
        earned.astype('d'),
        max(int(round(float(taper) / max(spacing, 1e-9))), 1), closed)


def _smoothed(values: np.ndarray, window: int, closed: bool) -> np.ndarray:
    """A moving average, wrapped for a circuit and edge-held for a road."""
    if window <= 1:
        return values
    padded = np.pad(values, window, mode='wrap' if closed else 'edge')
    kernel = np.ones(2 * window + 1) / float(2 * window + 1)
    found: np.ndarray = np.convolve(padded, kernel, mode='same')[window:-window]
    return found


def _runs_of(flag: np.ndarray, closed: bool) -> list[tuple[int, int]]:
    """``(first, last)`` inclusive for each contiguous stretch that is set.

    A circuit's last stretch runs on into its first, so the two are joined into
    one rather than counted as a pair that each fall short of a length they
    together clear.
    """
    found: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(flag):
        if on and start is None:
            start = index
        elif not on and start is not None:
            found.append((start, index - 1))
            start = None
    if start is not None:
        found.append((start, len(flag) - 1))
    if (closed and len(found) > 1 and found[0][0] == 0
            and found[-1][1] == len(flag) - 1):
        # The stretch across the join is one stretch. Kept as two spans of the
        # array, since the caller indexes it, but the shorter is dropped into
        # the longer's place so neither is measured on its own.
        first, last = found.pop(0), found.pop()
        found.append((last[0], len(flag) - 1))
        found.append((first[0], first[1]))
    return found


def _tapered(flag: np.ndarray, window: int, closed: bool) -> np.ndarray:
    """A 0/1 flag opened into a ramp over ``window`` points either side.

    Full across the whole of what was flagged, nothing ``window`` points clear
    of it, and a straight line between -- so a widened stretch is exactly its
    full width where it was earned and exactly the road's own width away from
    it, with the opening out spread evenly over the approach.

    A ramp rather than an average of the flag: averaging reaches half its height
    at the edge of what was flagged and the rest of the way in one step, which
    for a lane is a lane that begins in mid-air.
    """
    if window <= 1 or not len(flag) or not np.any(flag > 0):
        return flag
    # How many points from each place to the nearest flagged one, swept in from
    # both ends: two passes, and no point is further than the road is long.
    away = np.where(flag > 0, 0.0, float(len(flag)))
    for _pass in range(2 if closed else 1):
        for index in range(1, len(away)):
            away[index] = min(away[index], away[index - 1] + 1.0)
        if closed:
            away[0] = min(away[0], away[-1] + 1.0)
        for index in range(len(away) - 2, -1, -1):
            away[index] = min(away[index], away[index + 1] + 1.0)
        if closed:
            away[-1] = min(away[-1], away[0] + 1.0)
    found: np.ndarray = np.clip(1.0 - away / float(window), 0.0, 1.0)
    return found


def _sight_clearing(radius: np.ndarray, speed: np.ndarray, caution: float,
                    least: float, most: float, from_eye: float) -> np.ndarray:
    """How far from the centreline the trees go, so a driver can see round.

    A line of sight past an obstruction ``clear`` metres to the side of a bend
    of radius *r* reaches about ``sqrt(8 * r * clear)`` round it, so the offset
    that buys a given sight distance is that read the other way -- and the
    corridor is measured from the centreline rather than from the driver, so it
    is ``from_eye`` further out again. A straight -- an enormous radius -- asks
    for nothing, which is right: there is nothing in the way of seeing down it.
    """
    wanted = np.array([stopping_distance(float(each) * float(caution))
                       for each in speed])
    with np.errstate(divide='ignore', invalid='ignore'):
        needed = wanted * wanted / (8.0 * np.maximum(radius, 1e-9))
    found: np.ndarray = np.clip(
        np.nan_to_num(needed, nan=0.0) + float(from_eye),
        float(least), float(most))
    return found
