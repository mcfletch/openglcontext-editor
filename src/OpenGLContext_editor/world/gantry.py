"""Where a circuit's start/finish gantry stands, taken from the road itself.

A lap begins where the centreline does, so nothing here is authored: the gantry
takes the crown at that station, spans the running surface, and turns to face
along the road. The one thing it has to look up is the ground, because the legs
stand off either side of the carriageway and the land under them is rarely level
with the tarmac.

The gantry itself -- the legs, the beam, the banner and the painted line -- is
the engine's: :mod:`OpenGLContext.scenegraph.gantry`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.gantry import GantryProfile

__all__ = ['StartFinish', 'start_finish', 'FOOTING']

#: How far below its own ground each foot is sunk, in metres. The drop is
#: measured against the height function the world was designed with, and a
#: terrain tile a hundred metres from the camera is a coarser surface than that.
#: A foot resting exactly on the design height shows daylight under it as soon
#: as the tile beneath coarsens; a buried one never does.
FOOTING = 0.25


@dataclass(frozen=True)
class StartFinish:
    """A gantry in the world: where it stands, which way, and how wide.

    ``position`` is the road's crown at the line and ``yaw`` the turn that puts
    the prototype -- beam along X, road along Z -- onto the road. ``span`` is
    the distance between the leg centres, ``width`` how far the painted line
    reaches across the carriageway, and ``crossfall`` the camber it is painted
    on. ``drops`` is how far below the road surface each leg's own ground lies,
    left leg first.
    """

    position: Any
    yaw: float
    span: float
    width: float
    crossfall: float = 0.0
    drops: tuple[float, float] = (0.0, 0.0)

    def __repr__(self) -> str:
        return 'StartFinish(%s, %.0fm wide)' % (
            ', '.join('%.1f' % v for v in self.position), self.span)


def start_finish(path: Any, profile: GantryProfile | None = None,
                 ground: Any = None, station: float = 0.0,
                 footing: float = FOOTING) -> StartFinish:
    """Where the gantry marking the lap belongs on this road.

    ``path`` is a :class:`~OpenGLContext_editor.world.road.RoadPath`;
    ``station`` is how far along it the line is drawn, which for a circuit is
    where the lap begins. ``ground`` is the land the legs are founded on;
    without it they stand at the height of the road they span, which is right
    wherever the road is on the ground.
    """
    profile = profile or GantryProfile()
    line = np.asarray(path.points, dtype='d').reshape(-1, 3)
    stations = np.asarray(path.stations, dtype='d')
    index = int(np.clip(np.searchsorted(stations, float(station)), 0,
                        len(line) - 1))
    forward = line[min(index + 1, len(line) - 1)] - line[max(index - 1, 0)]
    yaw = math.atan2(float(forward[0]), float(forward[2]))
    road = path.profile
    span = road.carriageway_width + 2.0 * (road.shoulder_width + profile.margin)
    at = line[index]
    return StartFinish(position=at, yaw=yaw, span=float(span),
                       width=float(road.carriageway_width),
                       crossfall=float(road.crossfall),
                       drops=_drops(at, yaw, span, ground, footing))


def _drops(at: np.ndarray, yaw: float, span: float, ground: Any,
           footing: float) -> tuple[float, float]:
    """How far below the road each foot has to reach, left leg first.

    Ground *above* the road counts as level: a leg on a bank the road is cut
    into starts at the road, not part-way up the hillside, or the beam over the
    carriageway hangs from a stump.
    """
    beam = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    found = []
    for side in (-1.0, 1.0):
        foot = at + beam * (side * span / 2.0)
        if ground is None:
            below = 0.0
        else:
            level = float(np.asarray(ground(foot[0], foot[2]),
                                     dtype='d').ravel()[0])
            below = max(float(at[1]) - level, 0.0)
        found.append(below + float(footing))
    return (found[0], found[1])
