"""Axis-aligned bounds, and the frame change a 3D Tiles bounding volume needs.

The engine measures geometry in a Y-up world. 3D Tiles states a tile's bounding
volume in the tile's own Z-up frame, and the runtime rotates it back on the way
in (``OpenGLContext.loaders.tiles3d.tileset``). :class:`BoundingBox` holds the
Y-up box the baker works in and hands out the Z-up ``box``/``sphere`` arrays a
tileset carries, so the quarter turn is written once.

Source: OGC 3D Tiles 1.1, §"Bounding volumes" — ``box`` is twelve numbers, a
centre followed by three half-axis vectors, and ``sphere`` is a centre and a
radius.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

#: Anything numpy will read as three coordinates: a tuple, a list, an array.
Point = npt.ArrayLike
Vector = npt.NDArray[np.float64]

#: A half-axis of exactly zero makes a tile a plane, and the runtime's
#: point-to-box distance then reports zero for any camera directly above it --
#: the tile never falls out of view and never stops refining. Flat content
#: (water, a road deck, a level terrace) is common enough that the floor is
#: applied to every axis rather than left to each caller.
MINIMUM_HALF_EXTENT = 1e-3


class BoundingBox:
    """An axis-aligned box in the engine's Y-up world frame.

    Constructed from two corners, from points, or by joining boxes with ``|``.
    Instances are treated as read-only; every operation returns a new box.
    """

    __slots__ = ('minimum', 'maximum')

    minimum: Vector
    maximum: Vector

    def __init__(self, minimum: Point, maximum: Point) -> None:
        self.minimum = np.asarray(minimum, dtype='d')
        self.maximum = np.asarray(maximum, dtype='d')

    # -- construction ----------------------------------------------------------

    @classmethod
    def of_points(cls, points: npt.ArrayLike) -> BoundingBox | None:
        """The box enclosing an (N,3) array of points, or None if there are none."""
        array = np.asarray(points, dtype='d')
        if not len(array):
            return None
        return cls(array.min(axis=0), array.max(axis=0))

    @classmethod
    def joined(cls, boxes: Iterable[BoundingBox | None]) -> BoundingBox | None:
        """The box enclosing every box given, or None if there are none."""
        result: BoundingBox | None = None
        for box in boxes:
            result = box if result is None else (result | box)
        return result

    @classmethod
    def centred(cls, center: Point, half: Point) -> BoundingBox:
        """The box of a given centre and half-extent."""
        c = np.asarray(center, dtype='d')
        h = np.asarray(half, dtype='d')
        return cls(c - h, c + h)

    # -- measurements ----------------------------------------------------------

    @property
    def center(self) -> Vector:
        middle: Vector = (self.minimum + self.maximum) * 0.5
        return middle

    @property
    def half(self) -> Vector:
        extent: Vector = (self.maximum - self.minimum) * 0.5
        return extent

    @property
    def size(self) -> Vector:
        extent: Vector = self.maximum - self.minimum
        return extent

    @property
    def diagonal(self) -> float:
        """The corner-to-corner distance, the natural scale of a tile's content."""
        return float(np.linalg.norm(self.size))

    def contains(self, point: Point) -> bool:
        """Whether a point lies inside, counting both faces of each axis."""
        p = np.asarray(point, dtype='d')
        return bool(np.all(p >= self.minimum) and np.all(p <= self.maximum))

    # -- derivation ------------------------------------------------------------

    def __or__(self, other: BoundingBox | None) -> BoundingBox:
        if other is None:
            return self
        return BoundingBox(np.minimum(self.minimum, other.minimum),
                           np.maximum(self.maximum, other.maximum))

    def expanded(self, margin: float) -> BoundingBox:
        """The box grown by ``margin`` on every side."""
        return BoundingBox(self.minimum - margin, self.maximum + margin)

    def with_height(self, low: float, high: float) -> BoundingBox:
        """The same footprint, with the vertical extent replaced."""
        return BoundingBox((self.minimum[0], low, self.minimum[2]),
                           (self.maximum[0], high, self.maximum[2]))

    def __repr__(self) -> str:
        return "BoundingBox(%s, %s)" % (list(self.minimum), list(self.maximum))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return bool(np.array_equal(self.minimum, other.minimum)
                    and np.array_equal(self.maximum, other.maximum))

    def __hash__(self) -> int:
        return hash((tuple(self.minimum), tuple(self.maximum)))

    # -- the tileset's frame ---------------------------------------------------

    def tiles_box(self) -> list[float]:
        """This box as a 3D Tiles ``box`` volume: centre then three half-axes.

        The tile frame is Z-up, so the Y-up box turns a quarter: world +Y becomes
        tile +Z and world +Z becomes tile -Y.
        """
        c, h = self.center, np.maximum(self.half, MINIMUM_HALF_EXTENT)
        return [float(c[0]), float(-c[2]), float(c[1]),
                float(h[0]), 0.0, 0.0,
                0.0, float(h[2]), 0.0,
                0.0, 0.0, float(h[1])]

    def tiles_sphere(self) -> list[float]:
        """This box as a 3D Tiles ``sphere`` volume enclosing it, in the tile frame."""
        c, h = self.center, np.maximum(self.half, MINIMUM_HALF_EXTENT)
        return [float(c[0]), float(-c[2]), float(c[1]), float(np.linalg.norm(h))]
