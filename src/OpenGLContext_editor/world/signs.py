"""What a road warns about, worked out from the road itself.

A generated road already knows what it is about to do. The alignment carries its
own curvature and its own grade, and the structures along it are written down,
so *which* sign belongs *where* is derivable rather than authored -- which is
most of the point of generating a road instead of drawing one. A designer who
disagrees adds or removes a :class:`Warning`; nobody has to place the ordinary
ones by hand.

A sign warns of a hazard a driver could not otherwise see in time, so it stands
a **stopping distance** before it: far enough to act on, near enough to be about
this hazard and not the next. What counts as a hazard is measured against the
design speed, because there is always some speed at which any corner is too
tight and the design speed is the answer to which corners count.

The object itself -- the post, the plate and the symbol on it -- is the engine's:
:mod:`OpenGLContext.scenegraph.roadsigns`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.road import (
    GRAVITY,
    advisory_speed,
    sweep_frames,
)
from OpenGLContext.scenegraph.roadsigns import LIMIT, SignFace, SignProfile

from OpenGLContext_editor.world.route import cornering_radius
from OpenGLContext_editor.world.structures import Op

__all__ = ['Warning', 'Placement', 'warn_of', 'sign_placements',
           'stopping_distance', 'LIMIT_SPACING', 'POSTED_LIMIT']


#: How long a driver takes to react, in seconds, and how hard a car brakes, as a
#: fraction of gravity. Together they give the distance a warning needs: a sign
#: a driver cannot act on before reaching what it warns of is decoration.
REACTION_SECONDS = 1.5
BRAKING = 0.55

#: How much tighter than the design corner a bend has to be before it is worth a
#: sign. A road held to its design radius is full of corners at exactly that
#: radius, and a sign at every one of them is a sign at none of them.
BEND_MARGIN = 0.85

#: How much the gradient has to swing through a dip or over a crest before it is
#: worth a sign, and how far either side of the turning point that swing is
#: measured over, in metres. A road rounded off for its design speed has no crest
#: that launches a car, but it still has dips and brows a driver wants to know
#: are coming -- and a valley two kilometres wide is scenery, not a dip, which is
#: what the window is for.
GRADE_SWING = 0.09
VERTICAL_WINDOW = 160.0

#: How far apart two hazards have to be before they are two signs, in metres. A
#: sign every twenty metres is a sign nobody reads; two bends inside this become
#: one double-bend.
SEPARATION = 220.0

#: How far apart the speed limit is repeated along a road, in metres. Far
#: enough that it is not clutter, near enough that a driver joining a road, or
#: coming out of a bore, is told what it is inside a minute of driving.
LIMIT_SPACING = 1500.0

#: How far a limit sign moves at a time while it is looking for room, in metres.
#: Short enough that it stops at the first place it fits, long enough that
#: finding a place on a kilometre of road is a few dozen comparisons.
LIMIT_STEP = 20.0

#: How far a limit sign keeps from a warning, in metres. Less than two warnings
#: keep from each other: a warning is a hazard to act on and wants the road to
#: itself either side of it, and a number on a white plate is a smaller thing to
#: read. At the design speed this is still nearly three seconds apart.
LIMIT_CLEARANCE = SEPARATION / 2.0

#: What a road is posted at unless it is told otherwise, in km/h. A decision
#: rather than a measurement -- which is the difference between this and the
#: number on a bend's tab -- so it is told to the road rather than worked out
#: from it.
#:
#: Eighty on a two-lane road through forest, which is what this one is: it is
#: also the speed everything else on the road is doing, and a racer's lap is
#: made of the difference between that and what they are doing.
POSTED_LIMIT = 80

#: Which warning survives when two fall within :data:`SEPARATION` of each other,
#: highest first. A driver about to go into the dark and round a bend needs
#: telling about the dark.
PRECEDENCE = ('tunnel', 'double-bend', 'bend-left', 'bend-right', 'dip', 'crest')

#: How far apart two bends turning opposite ways may be and still be one
#: hazard -- a double bend -- in metres.
LINKED = 120.0

#: How short a stretch of gentler curve inside a bend is still the same bend, in
#: metres. A radius that wobbles over the limit and back is one corner to drive,
#: and without this a long constant bend is signed four times.
BEND_GAP = 55.0

#: How far the road has to turn the other way, in metres, before that is a
#: change of hand rather than a wobble. A left running straight into a right is
#: one unbroken tight stretch, so the reversal has to be found inside it -- and
#: a route slid sideways to follow the contours wanders the other way for a few
#: metres all the time. At a design speed of forty metres a second this is under
#: two seconds of opposite lock, which is not a corner anybody names.
REVERSAL = 70.0

#: Which side of the road signs stand on: +1 for traffic that keeps right, -1
#: for traffic that keeps left.
NEARSIDE = 1


def stopping_distance(speed: float, reaction: float = REACTION_SECONDS,
                      braking: float = BRAKING) -> float:
    """How far a car covers before it can be stopped, in metres.

    The distance travelled while the driver reacts plus the distance under the
    brakes. It is what a sign is placed at: closer and there is nothing to be
    done about it, much further and it is about somewhere else.
    """
    speed = max(float(speed), 0.0)
    return float(speed * reaction + speed * speed / (2.0 * braking * GRAVITY))


@dataclass(frozen=True)
class Warning:
    """One sign: where it stands, what it says, and where that is about.

    ``station`` and ``hazard`` are distances along the road in metres -- where
    the sign is and where the thing it is about begins. ``side`` is +1 for the
    right-hand edge of the road in the direction of travel and -1 for the left.

    ``speed`` is in km/h: under a bend it is the tab saying how fast that bend
    is worth, and on a :data:`~OpenGLContext.scenegraph.roadsigns.LIMIT` it is
    the number the sign is. Zero for a sign that says nothing about speed.
    """

    station: float
    hazard: float
    kind: str
    side: int = NEARSIDE
    speed: int = 0

    def __repr__(self) -> str:
        return 'Warning(%s%s at %.0fm, of %.0fm)' % (
            self.kind, ' %d' % self.speed if self.speed else '',
            self.station, self.hazard)

    @property
    def face(self) -> SignFace:
        """The sign this is, as the engine builds one."""
        return SignFace(self.kind, self.speed)


@dataclass(frozen=True)
class Placement:
    """A sign in the world: where its foot is, which way it looks, and which."""

    position: Any
    yaw: float
    kind: str
    speed: int = 0

    def __repr__(self) -> str:
        return 'Placement(%s at %s)' % (
            self.face, ', '.join('%.1f' % v for v in self.position))

    @property
    def face(self) -> SignFace:
        """The sign this is, as the engine builds one."""
        return SignFace(self.kind, self.speed)


def warn_of(path: Any, design_speed: float, side: int = NEARSIDE,
            separation: float = SEPARATION,
            limit: int = POSTED_LIMIT) -> list[Warning]:
    """Every sign a road wants, in order along it.

    ``path`` is a :class:`~OpenGLContext_editor.world.road.RoadPath`; the
    curvature, the grade and the structures all come off it. ``limit`` is what
    the road is posted at, in km/h, and 0 leaves it unposted.
    """
    line = np.asarray(path.points, dtype='d').reshape(-1, 3)
    if len(line) < 5:
        return []
    stations = np.asarray(path.stations, dtype='d')
    found = (_bends(line, stations, design_speed,
                    np.asarray(path.bank, dtype='d'))
             + _grades(line, stations)
             + _bores(path))
    ahead = stopping_distance(design_speed)
    placed = [Warning(station=max(float(at - ahead), 0.0), hazard=float(at),
                      kind=kind, side=int(side), speed=speed)
              for at, kind, speed in sorted(found)]
    warnings = _thinned(placed, separation)
    return sorted(warnings + _limits(stations, warnings, side, limit, separation),
                  key=lambda one: one.station)


def _limits(stations: np.ndarray, warnings: list[Warning], side: int,
            limit: int, separation: float) -> list[Warning]:
    """The speed limit, repeated along the road.

    A limit sign is not about a hazard, so it stands where the road has room
    for one rather than where anything is happening: on the interval, and never
    inside ``separation`` of a warning, because two plates a driver reads as one
    sign is a driver who has read neither.

    **A blocked one moves rather than going away.** Since it is not about a
    place, the sign slides along the road until there is room for it -- a
    circuit with a bend warning every few hundred metres would otherwise be
    posted nowhere, which is worse than being posted a hundred metres from
    where the interval fell.
    """
    if limit <= 0:
        return []
    length = float(stations[-1])
    out: list[Warning] = []
    for step in range(1, int(length // LIMIT_SPACING) + 1):
        at = _room_for_a_limit(step * LIMIT_SPACING, warnings, out,
                               min(separation, LIMIT_CLEARANCE), length)
        if at is None:
            continue
        out.append(Warning(station=at, hazard=at, kind=LIMIT, side=int(side),
                           speed=int(limit)))
    return out


def _room_for_a_limit(wanted: float, warnings: list[Warning],
                      placed: list[Warning], separation: float,
                      length: float) -> float | None:
    """Where near ``wanted`` a limit sign fits, or None for nowhere near it.

    Nearest first, either side of where the interval fell, so a sign moves as
    little as it has to. It gives up at half an interval, past which it would be
    standing where the next one belongs.
    """
    for step in np.arange(0.0, LIMIT_SPACING / 2.0, LIMIT_STEP):
        for at in ((wanted - step, wanted + step) if step else (wanted,)):
            if not 0.0 <= at <= length:
                continue
            if all(abs(at - one.station) >= separation
                   for one in list(warnings) + placed):
                return float(at)
    return None


def sign_placements(path: Any, warnings: Sequence[Warning],
                    profile: SignProfile | None = None,
                    ground: Any = None) -> list[Placement]:
    """Where each warning's sign stands in the world, and which way it looks.

    The post stands the road's half-width plus the profile's own offset out to
    ``side``, facing back down the road at the traffic it is for. ``ground`` is
    the land it is founded on; without it the sign stands at the height of the
    road beside it, which is right wherever the road is on the ground.
    """
    profile = profile or SignProfile()
    line = np.asarray(path.points, dtype='d').reshape(-1, 3)
    stations = np.asarray(path.stations, dtype='d')
    right, _up = sweep_frames(line)
    out = float(path.profile.total_width) / 2.0 + profile.offset
    placed = []
    for warning in warnings:
        index = int(np.clip(np.searchsorted(stations, warning.station), 0,
                            len(line) - 1))
        at = line[index] + right[index] * (out * warning.side)
        if ground is not None:
            at = at.copy()
            at[1] = float(np.asarray(ground(at[0], at[2]), dtype='d').ravel()[0])
        forward = line[min(index + 1, len(line) - 1)] - line[max(index - 1, 0)]
        placed.append(Placement(position=at, kind=warning.kind,
                                speed=warning.speed,
                                yaw=_facing_back(forward)))
    return placed


def _facing_back(forward: np.ndarray) -> float:
    """The yaw that turns a prototype facing -Z round to meet oncoming traffic."""
    return float(np.arctan2(-forward[0], -forward[2]) + np.pi)


def _bends(line: np.ndarray, stations: np.ndarray, design_speed: float,
           bank: np.ndarray) -> list[tuple[float, str, int]]:
    """Where the road turns more tightly than it is meant to be driven.

    A run of points below the limit is one bend, however many points it is, and
    the sign is about where it *starts*: that is what a driver is arriving at.
    Two bends close together are one double-bend, because that is what they are
    to drive and two plates fifty metres apart tell nobody anything.

    Each carries the speed its own tightest radius allows
    (:func:`~OpenGLContext.scenegraph.road.advisory_speed`), which is what goes
    on the tab under the plate: the road knows how tightly it turns, so how fast
    that is worth is derivable in the same way the symbol is.

    **How far the bend leans counts as much as how tight it is.** A
    superelevated corner is faster than the same corner flat, and both the
    decision to sign it at all and the number on the tab are taken with the lean
    in hand: a plate warning a driver off a bend the road is holding them
    through is a plate they learn to ignore.
    """
    steepest = float(np.abs(bank).max()) if len(bank) else 0.0
    limit = cornering_radius(design_speed, bank=steepest) / max(BEND_MARGIN, 1e-6)
    if limit <= 0.0:
        return []
    radius, turn = _curvature(line)
    tight = _closed_up(radius < limit, stations, BEND_GAP)
    found = []
    for first, last in _runs(tight):
        hands = _hands(turn[first:last + 1], stations[first:last + 1])
        for index, (start, hand) in enumerate(hands):
            over = (first + start,
                    first + (hands[index + 1][0] if index + 1 < len(hands)
                             else last - first + 1))
            found.append((float(stations[first + start]),
                          'bend-left' if hand > 0 else 'bend-right',
                          advisory_speed(
                              float(radius[over[0]:over[1]].min()),
                              bank=float(np.abs(
                                  bank[over[0]:over[1]]).min()))))
    return _linked(found, stations)


def _hands(turn: np.ndarray, stations: np.ndarray) -> list[tuple[int, float]]:
    """Which way each part of one tight stretch goes, as (first index, hand).

    A left running straight into a right is one unbroken stretch of tight
    radius, so the change of hand has to be found inside it -- and a sign flip
    lasting a few metres is the line wobbling, not the road turning back.
    """
    out: list[tuple[int, float]] = []
    for first, last in _runs(turn > 0) + _runs(turn <= 0):
        if stations[last] - stations[first] >= REVERSAL or not out:
            out.append((first, 1.0 if turn[first] > 0 else -1.0))
    return sorted(out)


def _closed_up(flags: np.ndarray, stations: np.ndarray,
               gap: float) -> np.ndarray:
    """Fill runs of False shorter than ``gap`` metres between two True runs."""
    found = np.asarray(flags, dtype=bool).copy()
    for first, last in _runs(~found):
        if first == 0 or last == len(found) - 1:
            continue
        if stations[last] - stations[first] < gap:
            found[first:last + 1] = True
    return found


def _linked(found: list[tuple[float, str, int]],
            stations: np.ndarray) -> list[tuple[float, str, int]]:
    """Bends close enough together to be one warning, merged into one.

    Two turning *opposite* ways within :data:`LINKED` metres are a double bend,
    which is the thing that sign means. Two the same way are one corner to
    drive, however the geometry happens to break them up, and a second plate
    saying the same thing tells nobody anything.
    """
    out: list[tuple[float, str, int]] = []
    for at, kind, speed in found:
        if out and at - out[-1][0] < LINKED:
            # One sign for both, and the slower of the two numbers: a driver
            # told 80 who then meets the half of it that is worth 50 has been
            # told the wrong thing.
            out[-1] = (out[-1][0],
                       'double-bend' if kind != out[-1][1] else out[-1][1],
                       min(one for one in (speed, out[-1][2]) if one)
                       if speed or out[-1][2] else 0)
            continue
        out.append((at, kind, speed))
    return out


def _grades(line: np.ndarray, stations: np.ndarray
            ) -> list[tuple[float, str, int]]:
    """Where the road bottoms out or tops over, sharply enough to be worth it.

    A dip and a crest are *turning points*: the grade reverses. Naming them by
    the local curvature instead gets three signs out of one dip, because the
    brows either side of it curve the other way and are as real as the bottom
    is; naming them by where the grade changes sign gets the one hazard a
    driver would name. How much of a hazard is the swing in gradient through
    it, measured over a window, so a valley two kilometres wide is not a dip.
    """
    middles = (stations[:-1] + stations[1:]) / 2.0
    grade = np.diff(line[:, 1]) / np.maximum(np.diff(stations), 1e-6)
    found = []
    for index in range(1, len(grade)):
        before, after = grade[index - 1], grade[index]
        if before == 0.0 or np.sign(before) == np.sign(after):
            continue
        at = float(middles[index])
        near = np.abs(middles - at) <= VERTICAL_WINDOW
        if float(grade[near].max() - grade[near].min()) < GRADE_SWING:
            continue
        found.append((at, 'dip' if after > before else 'crest', 0))
    return found


def _bores(path: Any) -> list[tuple[float, str, int]]:
    """Where the road goes into the dark.

    Only tunnels: a driver needs no telling that a road is on a bridge, and a
    causeway is a road.
    """
    return [(start, 'tunnel', 0) for kind, start, _end in path.structure_runs()
            if kind is Op.TUNNEL]


def _curvature(line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The radius the road turns at each point, and which way, in plan.

    The circle through each point and its two neighbours. Positive turn is to
    the left, looking along the direction of travel.
    """
    plan = line[:, [0, 2]]
    before, here, after = plan[:-2], plan[1:-1], plan[2:]
    a = np.linalg.norm(here - before, axis=1)
    b = np.linalg.norm(after - here, axis=1)
    c = np.linalg.norm(after - before, axis=1)
    cross = ((here[:, 0] - before[:, 0]) * (after[:, 1] - before[:, 1])
             - (here[:, 1] - before[:, 1]) * (after[:, 0] - before[:, 0]))
    area = np.abs(cross) / 2.0
    radius = np.where(area > 1e-9, a * b * c / np.maximum(4.0 * area, 1e-12),
                      np.inf)
    # The ends take their neighbour's answer, which is where a road is straight
    # anyway and stops a spurious bend at every open end.
    return (np.concatenate([[radius[0]], radius, [radius[-1]]]),
            np.concatenate([[cross[0]], cross, [cross[-1]]]))


def _runs(flags: Any) -> list[tuple[int, int]]:
    """The (first, last) index of each run of True."""
    found = np.asarray(flags, dtype=bool)
    out: list[tuple[int, int]] = []
    start = None
    for index, on in enumerate(found):
        if on and start is None:
            start = index
        elif not on and start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(found) - 1))
    return out


def _thinned(placed: list[Warning], separation: float) -> list[Warning]:
    """One sign per cluster: the most important thing in it, placed earliest.

    Signs within ``separation`` of each other are one warning to a driver, and
    which of them survives is not "whichever came first in the list" -- a bend
    and a tunnel portal a hundred metres apart is a tunnel.
    """
    out: list[Warning] = []
    for warning in placed:
        if out and warning.station - out[-1].station < separation:
            speeds = [one for one in (out[-1].speed, warning.speed) if one]
            out[-1] = Warning(station=out[-1].station, hazard=warning.hazard,
                              kind=_together(out[-1].kind, warning.kind),
                              side=warning.side,
                              speed=min(speeds) if speeds else 0)
            continue
        out.append(warning)
    return out


def _together(first: str, second: str) -> str:
    """What one sign says when it has to stand for two hazards.

    Two bends turning opposite ways are a double bend, which is exactly what
    that sign means. Anything else is whichever matters more.
    """
    if {first, second} == {'bend-left', 'bend-right'}:
        return 'double-bend'
    return first if _rank(first) <= _rank(second) else second


def _rank(kind: str) -> int:
    return PRECEDENCE.index(kind) if kind in PRECEDENCE else len(PRECEDENCE)
