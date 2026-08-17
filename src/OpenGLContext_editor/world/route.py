"""Sliding a drawn plan onto ground a road can actually follow.

A line drawn across a landscape without regard for it climbs and drops wherever
the landscape does. Held afterwards to a grade a car can drive
(:func:`~OpenGLContext_editor.world.road.follow_terrain`), the alignment then
departs from the ground by whatever the difference was -- and over real relief
that means a viaduct or a bore for most of its length, which is a road *over* a
landscape rather than a road *through* one.

The answer is not to give up on the grade. It is to move the line.
:func:`ease_route` slides each point along its own contour, towards the height
its neighbours are at, which finds the route through the same country that the
ground itself supports: round the shoulder of a hill instead of over it, along a
valley instead of across it.

It is a *plan* operation, and deliberately separate from settling the profile:
what a designer drew is a shape and a place, and this keeps both -- no point
moves further than ``reach`` from where it was put -- while giving the shape the
ground under it a say.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

HeightFn = Callable[[Any, Any], Any]

__all__ = ['ease_route', 'hold_radius', 'least_radius',
           'cornering_radius', 'REACH', 'ROUNDS', 'RELAXATION', 'GRIP']

#: How far a point may end up from where it was drawn, in metres. Far enough to
#: go round a hill rather than over it; near enough that the circuit a designer
#: drew is still the circuit they get.
REACH = 220.0

#: How many rounds of sliding, and how much of each round's correction is taken.
#: Under-relaxed, because a point that jumps the whole way overshoots and the
#: line rings; hundreds of small steps settle where tens of large ones
#: oscillate, and each round is a handful of array operations.
ROUNDS = 500
RELAXATION = 0.35

#: How far apart the two samples are that measure the ground's slope across the
#: route, in metres. Wide enough to read the hillside rather than the hummocks
#: on it, which is the scale a road is routed at.
PROBE = 30.0

#: The steepest sideways slope worth stepping down, as rise over run. Where the
#: ground across the route is flatter than this there is no contour to follow
#: and the step would be enormous; the point stays where it is.
FLATTEST = 0.02

#: How far one round may move a point, in metres. A cap rather than a target: it
#: is what stops a step across nearly level ground from throwing the line into
#: the next valley before the smoothing can answer.
LONGEST_STEP = 25.0

#: How much of a car's weight is available sideways in a corner, as a fraction:
#: what a tyre on dry tarmac has to hold it on the line. It is what turns the
#: speed a road is designed for into the tightest corner it may have.
GRIP = 1.0

#: Standard gravity, m/s**2.
GRAVITY = 9.81

#: How much of a corner's excess is taken out per round, and how many rounds it
#: is given. Under-relaxed for the same reason the sliding is.
CORNER_RELAXATION = 0.35
CORNER_ROUNDS = 2000

#: How much the line is fair-smoothed each round, as a fraction of the way to
#: the average of a point's neighbours. What stops the sliding from folding the
#: line over itself where two neighbouring points find different contours --
#: enough to keep the line fair, and no more, because tension pulls the route
#: back towards the straight line it was trying to leave.
TENSION = 0.05


def cornering_radius(design_speed: float, grip: float = GRIP) -> float:
    """The tightest corner a road may have, in metres.

    A car of speed ``v`` on a corner of radius ``R`` needs ``v**2 / R`` of
    lateral acceleration to stay on it, and has ``grip * g`` to find it with.
    A road with no design speed has no limit -- there is always some speed at
    which any corner is too tight, and the answer to that is to say how fast the
    road is meant to be driven.
    """
    if design_speed <= 0 or grip <= 0:
        return 0.0
    return float(design_speed * design_speed / (grip * GRAVITY))


def least_radius(plan: Any, closed: bool = False) -> float:
    """The tightest corner in a plan, in metres, or infinity if it is straight.

    The circle through each point and its two neighbours: the radius the road
    actually turns at there.
    """
    line = np.asarray(plan, dtype='d').reshape(-1, 2)
    if len(line) < 3:
        return float('inf')
    before, after = _neighbours(line, closed)
    return float(np.min(_radius(before, line, after)))


def hold_radius(plan: Any, minimum: float, closed: bool = False,
                rounds: int = CORNER_ROUNDS,
                relaxation: float = CORNER_RELAXATION) -> np.ndarray:
    """Open out any corner in a plan tighter than ``minimum``.

    A point turning more sharply than that moves towards the chord between its
    neighbours, by a share of the way each round, until nothing needs to move --
    which spreads a kink into the curve that replaces it. An open route's ends
    stay where the designer put them.
    """
    line = np.asarray(plan, dtype='d').reshape(-1, 2).copy()
    if len(line) < 3 or minimum <= 0.0:
        return line
    for _round in range(int(rounds)):
        before, after = _neighbours(line, closed)
        chord = 0.5 * (before + after)
        tight = np.clip(1.0 - _radius(before, line, after) / minimum, 0.0, 1.0)
        if not tight.any():
            break
        step = (relaxation * tight)[:, None]
        if not closed:
            step[0] = step[-1] = 0.0
        line = line + (chord - line) * step
    return line


def ease_route(plan: Any, height_fn: HeightFn, reach: float = REACH,
               rounds: int = ROUNDS, relaxation: float = RELAXATION,
               closed: bool = False, minimum_radius: float = 0.0,
               spacing: float = 0.0) -> np.ndarray:
    """A drawn plan slid sideways onto ground a road can follow.

    ``plan`` is (N,2) XZ as it was drawn and the result is (N,2) in the same
    order, so it goes straight back to
    :func:`~OpenGLContext_editor.world.road.follow_terrain`. ``height_fn`` is
    the *undisturbed* ground: what the route has to live with, before any
    earthwork.

    ``spacing`` re-samples the plan to that interval first, and is what a caller
    holding a radius wants: a plan of points tens of metres apart is a polygon,
    and the road built along it turns through the whole of each corner at one
    vertex however gentle the polygon looks. Re-sampled to the spacing the road
    will be built at, the radius held is the radius driven.

    ``reach`` is how far a point may end up from where it was drawn. An open
    route keeps its ends exactly, because a designer put them somewhere on
    purpose -- a junction, a start line.

    ``minimum_radius`` holds the corners the sliding puts in to something a car
    can take -- see :func:`cornering_radius`. It is applied between rounds
    rather than at the end, because a corner opened out once is a corner the
    next round can close again; the two are projections onto sets that both
    contain the drawn line, so alternating them settles inside both.
    """
    drawn = np.asarray(plan, dtype='d').reshape(-1, 2)
    if spacing > 0.0 and len(drawn) >= 2:
        drawn = _resampled(drawn, spacing, closed)
    if len(drawn) < 3 or reach <= 0.0:
        return drawn.copy()
    line = drawn.copy()
    for _round in range(int(rounds)):
        line = _slide(line, drawn, height_fn, reach, relaxation, closed)
        if minimum_radius > 0.0:
            line = _held(hold_radius(line, minimum_radius, closed, rounds=1),
                         drawn, reach)
    if minimum_radius > 0.0:
        line = _held(hold_radius(line, minimum_radius, closed), drawn, reach)
    return line


def _slide(line: np.ndarray, drawn: np.ndarray, height_fn: HeightFn,
           reach: float, relaxation: float, closed: bool) -> np.ndarray:
    """One round: each point steps along its contour towards its neighbours."""
    before, after = _neighbours(line, closed)
    across = _across(after - before)
    height = np.asarray(height_fn(line[:, 0], line[:, 1]), dtype='d')
    wanted = 0.5 * (np.asarray(height_fn(before[:, 0], before[:, 1]), dtype='d')
                    + np.asarray(height_fn(after[:, 0], after[:, 1]), dtype='d'))

    # How steeply the ground falls across the route, from a wide pair of probes:
    # the sideways distance that changes this point's height by one metre.
    one = line + across * PROBE
    other = line - across * PROBE
    slope = (np.asarray(height_fn(one[:, 0], one[:, 1]), dtype='d')
             - np.asarray(height_fn(other[:, 0], other[:, 1]), dtype='d')) \
        / (2.0 * PROBE)
    movable = np.abs(slope) > FLATTEST
    step = np.zeros(len(line))
    step[movable] = np.clip(
        relaxation * (wanted[movable] - height[movable]) / slope[movable],
        -LONGEST_STEP, LONGEST_STEP)

    moved = line + across * step[:, None]
    # Fair the line before clamping, so a point that found a contour its
    # neighbours did not is pulled back into the run rather than left as a kink.
    moved = moved + TENSION * (0.5 * (before + after) - moved)
    if not closed:
        moved[0], moved[-1] = drawn[0], drawn[-1]
    return _held(moved, drawn, reach)


def _neighbours(line: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Each point's two neighbours along the route.

    An open route's ends have only one, and take themselves for the other, which
    leaves them where they are -- which is where they belong.
    """
    before, after = np.roll(line, 1, axis=0), np.roll(line, -1, axis=0)
    if not closed:
        before[0], after[-1] = line[0], line[-1]
    return before, after


def _across(along: np.ndarray) -> np.ndarray:
    """A unit vector across the route at each point, in plan."""
    length = np.linalg.norm(along, axis=1, keepdims=True)
    direction = np.divide(along, np.where(length > 1e-9, length, 1.0))
    return np.stack([-direction[:, 1], direction[:, 0]], axis=-1)


def _resampled(plan: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """A plan at an even spacing along its own length, closed if it is."""
    line = np.vstack([plan, plan[:1]]) if closed else plan
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    length = float(along[-1])
    if length <= 0.0:                            # pragma: no cover - degenerate
        return plan.copy()
    count = max(int(np.ceil(length / spacing)), 3)
    # A closed plan drops the repeated end: the ring is the points, and the
    # neighbour of the last is the first.
    wanted = (np.linspace(0.0, length, count, endpoint=False) if closed
              else np.linspace(0.0, length, count + 1))
    return np.stack([np.interp(wanted, along, line[:, axis])
                     for axis in range(2)], axis=-1)


def _radius(before: np.ndarray, here: np.ndarray,
            after: np.ndarray) -> np.ndarray:
    """The radius of the circle through each triple of points.

    ``abc / 4A`` for a triangle of sides a, b, c and area A. Three points in a
    line enclose no area and turn through no angle, which is a straight road and
    an infinite radius.
    """
    first, second = here - before, after - here
    span = after - before
    twice_area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    sides = (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
             * np.linalg.norm(span, axis=1))
    turning = twice_area > 1e-12
    found = np.full(len(here), np.inf)
    found[turning] = sides[turning] / (2.0 * twice_area[turning])
    return found


def _held(moved: np.ndarray, drawn: np.ndarray, reach: float) -> np.ndarray:
    """Pull anything that wandered further than ``reach`` back to the circle."""
    offset = moved - drawn
    distance = np.linalg.norm(offset, axis=1, keepdims=True)
    scale = np.where(distance > reach, reach / np.where(distance > 0, distance, 1.0),
                     1.0)
    return drawn + offset * scale
