"""Iso-height lines over a height field.

A plan view shaded for relief says where the land rises; contours say by how
much, and they are what a designer runs a road *along* when the road has to
hold a grade round a hillside. They are also what an iso-height snap needs to
be drawn against, so the two agree about where a hundred metres is.

The extraction is marching squares: each cell of the sampled grid contributes a
line segment across it wherever the level crosses, interpolated along the cell's
edges, and the segments are joined end to end into polylines. It is exact for a
field that is linear inside a cell and converges on the truth as the sampling
gets finer, which is what a *drawn* contour needs; nothing here is a
substitute for asking the height function itself where a particular elevation
is.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ['Contour', 'contour_levels', 'contours', 'contours_of']

#: How close two ends have to be to be the same point, as a fraction of a
#: cell. Segments are cut at cell edges, so the ends that should meet are the
#: same arithmetic and differ only in the last bits; anything looser would join
#: two contours that merely pass near each other.
JOIN_TOLERANCE = 1e-6


@dataclass
class Contour:
    """One elevation, and the lines the ground crosses it along.

    ``lines`` are ``(N,2)`` arrays of ``(x, z)`` in world metres. A line that
    comes back to where it started is a closed loop -- a hilltop or a basin --
    and one that does not runs off the edge of the ground.
    """

    elevation: float
    lines: list[np.ndarray] = field(default_factory=list)

    @staticmethod
    def closed(line: Any, tolerance: float = 1e-6) -> bool:
        """Whether a line comes back to where it started."""
        points = np.asarray(line, dtype='d')
        if len(points) < 3:
            return False
        return bool(np.linalg.norm(points[0] - points[-1]) <= tolerance)


def contour_levels(low: float, high: float, interval: float,
                   base: float = 0.0) -> np.ndarray:
    """The elevations to draw between two heights, at a spacing.

    Multiples of ``interval`` above ``base``, strictly inside the range: a
    contour exactly at the lowest point of the ground is a point, not a line.
    """
    interval = float(interval)
    if interval <= 0.0:
        raise ValueError("a contour interval must be a distance, not %r"
                         % (interval,))
    low, high = float(low), float(high)
    if not high > low:
        return np.zeros((0,), dtype='d')
    first = np.ceil((low - base) / interval)
    last = np.floor((high - base) / interval)
    if last < first:
        return np.zeros((0,), dtype='d')
    return base + interval * np.arange(first, last + 1.0)


#: Which pairs of cell edges a segment runs between, by case. The case is a
#: bitmask of which corners are below the level: 1 for (i, j), 2 for (i+1, j),
#: 4 for (i+1, j+1), 8 for (i, j+1). Edges are numbered the same way round: 0
#: is the (i, j)-(i+1, j) edge, 1 is (i+1, j)-(i+1, j+1), 2 is
#: (i, j+1)-(i+1, j+1) and 3 is (i, j)-(i, j+1).
#:
#: Cases 5 and 10 are the saddle: two opposite corners below and two above,
#: which can be read as two separate lines either way round. They are resolved
#: against the cell's own average, so the reading agrees with the field rather
#: than with a coin toss.
_CASES: dict[int, tuple[tuple[int, int], ...]] = {
    0: (), 15: (),
    1: ((3, 0),), 14: ((3, 0),),
    2: ((0, 1),), 13: ((0, 1),),
    3: ((3, 1),), 12: ((3, 1),),
    4: ((1, 2),), 11: ((1, 2),),
    6: ((0, 2),), 9: ((0, 2),),
    7: ((3, 2),), 8: ((3, 2),),
}

#: The saddle read each way: corners 1 and 4 below (case 5) joins the pairs one
#: way when the cell's middle is below the level and the other way when it is
#: above.
_SADDLE_SPLIT = ((3, 0), (1, 2))
_SADDLE_JOINED = ((3, 2), (0, 1))


def contours(heights: Any, min_x: float, max_x: float,
             min_z: float, max_z: float, interval: float,
             base: float = 0.0) -> list[Contour]:
    """The iso-height lines of a sampled height field.

    ``heights`` is a 2-D array sampled on a regular grid, ``heights[i, j]``
    being the ground at ``x = min_x + i * dx``, ``z = min_z + j * dz`` -- the
    layout ``numpy.meshgrid(..., indexing='ij')`` produces.
    """
    grid = np.asarray(heights, dtype='d')
    if grid.ndim != 2 or min(grid.shape) < 2:
        return []
    levels = contour_levels(float(grid.min()), float(grid.max()), interval,
                            base)
    if not len(levels):
        return []
    xs = np.linspace(float(min_x), float(max_x), grid.shape[0])
    zs = np.linspace(float(min_z), float(max_z), grid.shape[1])
    found = []
    for level in levels:
        segments = _segments(grid, xs, zs, float(level))
        # A level that passes exactly through a sample crosses its edges at
        # their ends, so the cell contributes a segment of no length. Dropping
        # them here keeps them out of the chaining, where a run of them would
        # otherwise come back as a "line" standing still at one point.
        if len(segments):
            reach = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            segments = segments[reach > 0.0]
        if not len(segments):
            continue
        found.append(Contour(elevation=float(level),
                             lines=_join(segments, xs, zs)))
    return found


def contours_of(height_fn: Callable[[Any, Any], Any], extent: float,
                interval: float, resolution: int = 129,
                base: float = 0.0,
                centre: tuple[float, float] = (0.0, 0.0)) -> list[Contour]:
    """The iso-height lines of a height function over a square of ground.

    ``extent`` is how many metres across that square is and ``resolution`` how
    many samples across it is read at -- the detail of the *drawn* line, and
    what it costs: the function is asked once, for the whole grid at a time.
    """
    half = float(extent) / 2.0
    xs = np.linspace(centre[0] - half, centre[0] + half, int(resolution))
    zs = np.linspace(centre[1] - half, centre[1] + half, int(resolution))
    grid_x, grid_z = np.meshgrid(xs, zs, indexing='ij')
    heights = np.asarray(height_fn(grid_x, grid_z), dtype='d')
    return contours(heights, xs[0], xs[-1], zs[0], zs[-1], interval, base)


def _crossings(grid: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the level cuts each grid edge, as a fraction along it.

    ``NaN`` where it does not cut, which is what keeps the answer the shape of
    the grid rather than a list whose indices have to be tracked back.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        along_x = (level - grid[:-1, :]) / (grid[1:, :] - grid[:-1, :])
        along_z = (level - grid[:, :-1]) / (grid[:, 1:] - grid[:, :-1])
    below = grid < level
    along_x = np.where(below[:-1, :] != below[1:, :], along_x, np.nan)
    along_z = np.where(below[:, :-1] != below[:, 1:], along_z, np.nan)
    return along_x, along_z


def _segments(grid: np.ndarray, xs: np.ndarray, zs: np.ndarray,
              level: float) -> np.ndarray:
    """Every segment the level makes, as an ``(N, 2, 2)`` array of ends."""
    along_x, along_z = _crossings(grid, level)
    below = grid < level
    case = (below[:-1, :-1].astype(np.intp)
            + below[1:, :-1].astype(np.intp) * 2
            + below[1:, 1:].astype(np.intp) * 4
            + below[:-1, 1:].astype(np.intp) * 8)
    middle = 0.25 * (grid[:-1, :-1] + grid[1:, :-1]
                     + grid[1:, 1:] + grid[:-1, 1:])
    pairs: list[np.ndarray] = []
    for value in np.unique(case):
        cells = np.argwhere(case == value)
        if not len(cells):
            continue
        if value in (5, 10):
            saddle = middle[cells[:, 0], cells[:, 1]] < level
            for joined, chosen in ((True, _SADDLE_JOINED),
                                   (False, _SADDLE_SPLIT)):
                which = cells[saddle == joined]
                if len(which):
                    pairs.append(_ends(which, chosen, along_x, along_z, xs, zs))
            continue
        edges = _CASES[int(value)]
        if edges:
            pairs.append(_ends(cells, edges, along_x, along_z, xs, zs))
    if not pairs:
        return np.zeros((0, 2, 2), dtype='d')
    return np.concatenate(pairs, axis=0)


def _ends(cells: np.ndarray, edges: Iterable[tuple[int, int]],
          along_x: np.ndarray, along_z: np.ndarray,
          xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """The world ends of one segment per cell, for each pair of edges."""
    made = [np.stack([_edge_point(cells, first, along_x, along_z, xs, zs),
                      _edge_point(cells, second, along_x, along_z, xs, zs)],
                     axis=1)
            for first, second in edges]
    return np.concatenate(made, axis=0)


def _edge_point(cells: np.ndarray, edge: int, along_x: np.ndarray,
                along_z: np.ndarray, xs: np.ndarray,
                zs: np.ndarray) -> np.ndarray:
    """Where the level cuts one edge of each of these cells, in world metres."""
    i, j = cells[:, 0], cells[:, 1]
    if edge == 0:                       # (i, j) -> (i+1, j)
        step = along_x[i, j]
        return np.stack([xs[i] + step * (xs[i + 1] - xs[i]), zs[j]], axis=1)
    if edge == 1:                       # (i+1, j) -> (i+1, j+1)
        step = along_z[i + 1, j]
        return np.stack([xs[i + 1], zs[j] + step * (zs[j + 1] - zs[j])], axis=1)
    if edge == 2:                       # (i, j+1) -> (i+1, j+1)
        step = along_x[i, j + 1]
        return np.stack([xs[i] + step * (xs[i + 1] - xs[i]), zs[j + 1]], axis=1)
    step = along_z[i, j]                # (i, j) -> (i, j+1)
    return np.stack([xs[i], zs[j] + step * (zs[j + 1] - zs[j])], axis=1)


def _join(segments: np.ndarray, xs: np.ndarray,
          zs: np.ndarray) -> list[np.ndarray]:
    """Chain segments end to end into polylines.

    Ends are matched on a quantised key rather than by distance: a segment ends
    on a cell edge and its neighbour begins on the same one, from the same
    arithmetic, so the two agree to the last bits. Rounding is to a fraction of
    a cell, which is fine enough that two contours passing near each other are
    never joined and coarse enough that the last bits do not separate them.
    """
    cell = min(abs(float(xs[1] - xs[0])), abs(float(zs[1] - zs[0]))) or 1.0
    grain = cell * JOIN_TOLERANCE

    def key(point: np.ndarray) -> tuple[int, int]:
        return (int(round(float(point[0]) / grain)),
                int(round(float(point[1]) / grain)))

    ends: dict[tuple[int, int], list[int]] = {}
    for index, segment in enumerate(segments):
        for point in segment:
            ends.setdefault(key(point), []).append(index)
    used = np.zeros(len(segments), dtype=bool)
    lines: list[np.ndarray] = []
    # Open chains first, from an end nothing else touches: starting a walk in
    # the middle of one would leave its two halves as two contours.
    starts = [index for index, segment in enumerate(segments)
              if any(len(ends[key(point)]) == 1 for point in segment)]
    for index in starts + list(range(len(segments))):
        if used[index]:
            continue
        lines.append(_walk(index, segments, ends, used, key))
    return lines


def _walk(index: int, segments: np.ndarray, ends: dict, used: np.ndarray,
          key: Callable[[np.ndarray], tuple[int, int]]) -> np.ndarray:
    """One polyline, from a segment outwards until it runs out or closes."""
    first, second = segments[index]
    if len(ends[key(first)]) == 1:
        points = [first, second]
    elif len(ends[key(second)]) == 1:
        points = [second, first]
    else:
        points = [first, second]
    used[index] = True
    while True:
        at = key(points[-1])
        following = [other for other in ends.get(at, ()) if not used[other]]
        if not following:
            return np.asarray(points, dtype='d')
        other = following[0]
        used[other] = True
        ahead, behind = segments[other]
        points.append(behind if key(ahead) == at else ahead)
