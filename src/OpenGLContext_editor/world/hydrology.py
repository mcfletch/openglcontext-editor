"""Water welling up at a point, and the river it makes.

A designer asking for a river does not want a blue stripe painted on the map;
they want water put down somewhere and the landscape to say where it goes. So
that is what this does: from a **spring**, follow the ground downhill until the
water reaches the waterline, runs off the edge of the world, or arrives
somewhere it cannot get out of -- which is where a lake is, and is an answer
rather than a failure.

Two things come out of a flow. The **path** is where the water runs, which the
editor draws and the designer moves. The :class:`Channel` is the bed it cuts,
which goes on the landscape's edit stack like any other edit: the river is then
part of the ground, the road crosses it as water rather than as a stripe, and
sculpting the land upstream reroutes it on the next settle.

Rivers that meet **merge**. A path that arrives on one already there stops and
adds its water to it, so the river below a junction carries both and cuts a
wider, deeper bed than either branch above it.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from OpenGLContext_editor.world.height import HeightEdit, register_edit

__all__ = ['Channel', 'FlowPath', 'Spring', 'channels_for', 'flow_from']

#: How far the water is walked at a time, in metres. Short enough to follow a
#: valley round its bends, long enough that a river across a four-kilometre
#: world is hundreds of points rather than thousands.
DEFAULT_STEP = 20.0

#: How many ways round the water looks for its next step. Every direction is
#: tried at once, so the cost is one call to the height function a step.
DIRECTIONS = 32

#: How far past a step the water will look to get out of a hollow, as a
#: multiple of the step. Ground is not a smooth ramp -- a real hillside is full
#: of small hollows -- and water that stopped in the first one it met would
#: never reach anything. Water fills a hollow until it spills, so the routing
#: looks past one; a hollow wider than this is a lake, which is an answer.
ESCAPE_REACH = 8

#: How near a path has to come to one already there to join it, as a multiple of
#: the step. Two rivers a step apart are one river.
JOIN_REACH = 1.5

#: How far the water may be walked before the routing gives up. A river that
#: has not reached the sea, the edge or a basin in this many steps is going
#: round something the step is too coarse to see.
STEP_LIMIT = 4000

#: How wide and how deep a bed the smallest river cuts, in metres, and how much
#: of each is added per doubling of what it carries. A stream is a notch and a
#: river carrying sixteen of them is something a road has to bridge.
BED_WIDTH = 9.0
BED_WIDTH_PER_DOUBLING = 5.0
BED_DEPTH = 2.0
BED_DEPTH_PER_DOUBLING = 1.1

#: How many samples of a channel are measured against its segments at a time.
#: The distance query is one array of samples against every segment, so this is
#: what keeps a whole tile's worth of ground from being multiplied out at once.
DISTANCE_CHUNK = 4096


@dataclass(frozen=True)
class Spring:
    """Where water wells up, in world metres."""

    at: tuple[float, float] = (0.0, 0.0)

    def to_json(self) -> dict[str, Any]:
        return {'at': [float(self.at[0]), float(self.at[1])]}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> Spring:
        at = document.get('at', (0.0, 0.0))
        return cls(at=(float(at[0]), float(at[1])))


@dataclass
class FlowPath:
    """Where one spring's water runs, and how much is in it along the way.

    ``ended`` says why the water stopped: ``water`` at the waterline, ``edge``
    off the world, ``basin`` somewhere it cannot get out of, ``joined`` on a
    river already there, or ``lost`` where the routing gave up.
    """

    points: np.ndarray
    flow: np.ndarray
    ended: str = 'edge'
    spring: Spring = field(default_factory=Spring)


#: The directions a step is tried in, as unit vectors, worked out once.
_ROUND = np.linspace(0.0, 2.0 * np.pi, DIRECTIONS, endpoint=False)
_WAYS = np.stack([np.cos(_ROUND), np.sin(_ROUND)], axis=1)


def _step_downhill(height_fn: Callable[[Any, Any], Any], x: float, z: float,
                   here: float, step: float,
                   reach: int) -> tuple[float, float] | None:
    """Where the water goes next, or None if it is somewhere it cannot leave.

    Every direction is tried at once and the lowest taken. If nothing a step
    away is lower, the same ring is tried further out, up to ``reach`` steps:
    that is water filling a hollow until it spills, and without it a river
    stops in the first dimple of a noisy hillside.

    The step that gets out of a hollow is taken whole rather than walked, so
    the river runs straight across the hollow it filled -- which is what a
    filled hollow is: a pond with the water leaving at the far side.
    """
    for out in range(1, int(reach) + 1):
        radius = step * out
        ahead_x = x + _WAYS[:, 0] * radius
        ahead_z = z + _WAYS[:, 1] * radius
        ahead = np.asarray(height_fn(ahead_x, ahead_z), dtype='d')
        lowest = int(ahead.argmin())
        if float(ahead[lowest]) < here:
            return (float(ahead_x[lowest]), float(ahead_z[lowest]))
    return None


def flow_from(height_fn: Callable[[Any, Any], Any], springs: Sequence[Spring],
              extent: float, step: float = DEFAULT_STEP,
              water_level: float | None = None,
              escape: int = ESCAPE_REACH) -> list[FlowPath]:
    """Route each spring's water downhill, merging the ones that meet.

    ``extent`` is how many metres across the world is, centred on the origin;
    water that leaves it has left. ``water_level`` is where the sea is, and
    None means the water runs until it reaches an edge or a basin.

    The springs are routed in the order given, and a path that comes within a
    step and a half of one already routed joins it. Which spring is the
    tributary and which the main river is therefore the order they were put
    down, which is what a designer building a system a spring at a time
    expects.
    """
    half = float(extent) / 2.0
    paths: list[FlowPath] = []
    for spring in springs:
        path = _walk(height_fn, spring, half, float(step), water_level,
                     int(escape), paths)
        paths.append(path)
    return paths


def _walk(height_fn: Callable[[Any, Any], Any], spring: Spring, half: float,
          step: float, water_level: float | None, escape: int,
          existing: list[FlowPath]) -> FlowPath:
    """One spring's water, followed until it stops."""
    x, z = float(spring.at[0]), float(spring.at[1])
    points = [(x, z)]
    ended = 'lost'
    for _ in range(STEP_LIMIT):
        joined = _join(existing, x, z, step)
        if joined is not None:
            ended = 'joined'
            break
        here = float(np.asarray(height_fn(np.asarray([x]),
                                          np.asarray([z])))[0])
        if water_level is not None and here <= water_level:
            ended = 'water'
            break
        ahead = _step_downhill(height_fn, x, z, here, step, escape)
        if ahead is None:
            # Nothing within reach is lower: the water is in a hollow it
            # cannot fill its way out of, which is where a lake is.
            ended = 'basin'
            break
        if max(abs(ahead[0]), abs(ahead[1])) >= half:
            # The step leaves the world: the water goes as far as the edge and
            # no further, so the river ends *on* the boundary rather than a
            # step past it.
            points.append(_at_edge((x, z), ahead, half))
            ended = 'edge'
            break
        x, z = ahead
        points.append((x, z))
    path = FlowPath(points=np.asarray(points, dtype='d'),
                    flow=np.ones(len(points), dtype='d'), ended=ended,
                    spring=spring)
    if ended == 'joined':
        _feed(existing, path.points[-1], step)
    return path


def _at_edge(here: tuple[float, float], ahead: tuple[float, float],
             half: float) -> tuple[float, float]:
    """Where a step out of the world crosses its boundary."""
    dx = ahead[0] - here[0]
    dz = ahead[1] - here[1]
    fraction = 1.0
    for at, delta in ((here[0], dx), (here[1], dz)):
        if delta > 1e-12:
            fraction = min(fraction, (half - at) / delta)
        elif delta < -1e-12:
            fraction = min(fraction, (-half - at) / delta)
    fraction = max(0.0, min(1.0, fraction))
    return (here[0] + dx * fraction, here[1] + dz * fraction)


def _join(existing: list[FlowPath], x: float, z: float,
          step: float) -> int | None:
    """The path this point has arrived on, or None."""
    reach = step * JOIN_REACH
    for index, path in enumerate(existing):
        if len(path.points) < 2:
            continue
        near = np.hypot(path.points[:, 0] - x, path.points[:, 1] - z).min()
        if near <= reach:
            return index
    return None


def _feed(existing: list[FlowPath], where: np.ndarray, step: float) -> None:
    """Add a tributary's water to every river below where it came in."""
    reach = step * JOIN_REACH
    for path in existing:
        if len(path.points) < 2:
            continue
        distance = np.hypot(path.points[:, 0] - where[0],
                            path.points[:, 1] - where[1])
        if distance.min() > reach:
            continue
        path.flow[int(distance.argmin()):] += 1.0


def channels_for(paths: Sequence[FlowPath],
                 width: float = BED_WIDTH,
                 depth: float = BED_DEPTH) -> list[Channel]:
    """The beds a set of flows cut, one per path that is long enough to cut one."""
    return [Channel(points=np.asarray(path.points, dtype='d'),
                    flow=np.asarray(path.flow, dtype='d'),
                    width=float(width), depth=float(depth))
            for path in paths if len(path.points) >= 2]


@register_edit
@dataclass
class Channel(HeightEdit):
    """A river bed cut along a path, wider and deeper where more water runs.

    ``width`` and ``depth`` are what the smallest river cuts, in metres; a
    river carrying twice as much cuts a fixed amount more of each, so a
    tributary is a notch and the river below a dozen junctions is something a
    road has to bridge rather than ford.
    """

    KIND: ClassVar[str] = 'channel'
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    flow: np.ndarray = field(default_factory=lambda: np.zeros(0))
    width: float = BED_WIDTH
    depth: float = BED_DEPTH

    # -- how big a river it is --------------------------------------------
    def _doublings(self) -> np.ndarray:
        """How many times over the smallest river this one is, along its length."""
        carried = np.maximum(np.asarray(self.flow, dtype='d'), 1.0)
        doublings: np.ndarray = np.log2(carried)
        return doublings

    def widths(self) -> np.ndarray:
        return self.width + BED_WIDTH_PER_DOUBLING * self._doublings()

    def depths(self) -> np.ndarray:
        return self.depth + BED_DEPTH_PER_DOUBLING * self._doublings()

    # -- the edit ----------------------------------------------------------
    def bounds(self) -> tuple[float, float, float, float]:
        line = np.asarray(self.points, dtype='d').reshape(-1, 2)
        if not len(line):
            return (0.0, 0.0, 0.0, 0.0)
        reach = float(self.widths().max())
        return (float(line[:, 0].min()) - reach, float(line[:, 1].min()) - reach,
                float(line[:, 0].max()) + reach, float(line[:, 1].max()) + reach)

    def delta(self, x: Any, z: Any, height: Any) -> np.ndarray:
        line = np.asarray(self.points, dtype='d').reshape(-1, 2)
        if len(line) < 2:
            return np.zeros(np.shape(x))
        flat_x = np.asarray(x, dtype='d').ravel()
        flat_z = np.asarray(z, dtype='d').ravel()
        cut = np.zeros(flat_x.shape)
        widths = self.widths()
        depths = self.depths()
        for start in range(0, len(flat_x), DISTANCE_CHUNK):
            piece = slice(start, start + DISTANCE_CHUNK)
            distance, at = _nearest(line, flat_x[piece], flat_z[piece])
            half = widths[at] / 2.0
            # A bed rather than a trench: the cut is deepest in the middle and
            # meets the land at the bank, so the ground either side is where it
            # was and a road crossing sees a valley rather than a step.
            across = np.clip(1.0 - distance / np.maximum(half, 1e-9), 0.0, 1.0)
            cut[piece] = -depths[at] * across * across * (3.0 - 2.0 * across)
        return cut.reshape(np.shape(x))

    # -- the file ----------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        line = np.asarray(self.points, dtype='d').reshape(-1, 2)
        return {'kind': self.KIND,
                'points': [[float(px), float(pz)] for px, pz in line],
                'flow': [float(value) for value in np.asarray(self.flow)],
                'width': float(self.width), 'depth': float(self.depth)}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> Channel:
        points = np.asarray(document.get('points', ()), dtype='d').reshape(-1, 2)
        flow = np.asarray(document.get('flow', ()), dtype='d')
        if len(flow) != len(points):
            flow = np.ones(len(points), dtype='d')
        return cls(points=points, flow=flow,
                   width=float(document.get('width', BED_WIDTH)),
                   depth=float(document.get('depth', BED_DEPTH)))


def _nearest(line: np.ndarray, x: np.ndarray,
             z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How far each sample is from a polyline, and which of its points is nearest.

    One array of samples against every segment at once. The caller chunks the
    samples, so what is multiplied out at any moment is bounded whatever size
    of ground is being asked about.
    """
    starts = line[:-1]
    ends = line[1:]
    along = ends - starts
    squared = np.maximum((along * along).sum(axis=1), 1e-12)
    offset_x = x[:, None] - starts[None, :, 0]
    offset_z = z[:, None] - starts[None, :, 1]
    fraction = np.clip((offset_x * along[None, :, 0]
                        + offset_z * along[None, :, 1]) / squared, 0.0, 1.0)
    closest_x = starts[None, :, 0] + along[None, :, 0] * fraction
    closest_z = starts[None, :, 1] + along[None, :, 1] * fraction
    distance = np.hypot(x[:, None] - closest_x, z[:, None] - closest_z)
    which = distance.argmin(axis=1)
    rows = np.arange(len(x))
    # The nearer end of the nearest segment says how big the river is there.
    at = which + (fraction[rows, which] > 0.5).astype(np.intp)
    return (distance[rows, which], np.minimum(at, len(line) - 1))
