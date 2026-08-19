"""Real elevation as a world's ground.

A designer who wants a particular valley does not want a landscape that looks
like one. This reads a published elevation product -- an SRTM height file --
and puts it under a world at a chosen point on the Earth, in metres, so a route
drawn on the map is a route across that ground.

Two questions have to be answered to do it, and both are arithmetic rather than
opinion: **where in the file** a given latitude and longitude is, and **how far
apart in metres** two latitudes and longitudes are. The facts behind both are
in [specs/ELEVATION-DATA.md](../../../specs/ELEVATION-DATA.md); the section
numbers below cite it, and nothing here was taken from a GIS tool's source.

Nothing fetches anything. A dataset is a file the designer supplies, named in
the project so the same ground comes back when it is opened again.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from OpenGLContext_editor.world.height import HeightBase, register_base

__all__ = ['VOID', 'DEMBase', 'ElevationGrid', 'LocalFrame', 'corner_from_name',
           'local_frame', 'read_hgt']

#: A sample with no data (spec 1.5). Left in the file as the smallest signed
#: 16-bit value, which is thirty-two kilometres below the sea if it is believed.
VOID = -32768

#: WGS 84 (spec 2.1): the ellipsoid the coordinates are given on.
EARTH_SEMI_MAJOR = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)

#: How a height file is named for the square it covers (spec 1.6).
_NAME = re.compile(r'^([NS])(\d{2})([EW])(\d{3})', re.IGNORECASE)


def corner_from_name(path: str) -> tuple[float, float]:
    """The ``(latitude, longitude)`` of a height file's south-west corner.

    From the file's name, which is where a `.hgt` says where it is: the file
    itself is samples and nothing else (spec 1.1, 1.6).
    """
    found = _NAME.match(os.path.basename(str(path)))
    if found is None:
        raise ValueError(
            "%r does not say which degree square it covers; a height file is "
            "named for its south-west corner, as in N47E008.hgt"
            % (os.path.basename(str(path)),))
    ns, lat, ew, lon = found.groups()
    south = float(lat) * (-1.0 if ns.upper() == 'S' else 1.0)
    west = float(lon) * (-1.0 if ew.upper() == 'W' else 1.0)
    return (south, west)


@dataclass
class ElevationGrid:
    """Elevations over a degree square, and how to read one out of it.

    ``samples`` is indexed ``[row, column]`` with **row 0 at the north edge**
    and column 0 at the west (spec 1.3), which is the order the file is in.
    """

    samples: np.ndarray
    south: float
    west: float
    #: How many degrees across the square is. One, for an SRTM tile (spec 1.4).
    span: float = 1.0

    @property
    def north(self) -> float:
        return self.south + self.span

    @property
    def east(self) -> float:
        return self.west + self.span

    def at(self, latitude: Any, longitude: Any) -> Any:
        """The elevation at a point, interpolated between the four samples
        around it.

        Outside the square the edge is repeated: a world framed near a tile's
        border would otherwise fall off the data, and an abyss at the edge of
        the map is worse than a shelf.
        """
        rows, columns = self.samples.shape
        # The first and last samples lie *on* the edges, so the spacing is one
        # span over (n - 1) rather than over n (spec 1.4).
        row = (self.north - np.asarray(latitude, dtype='d')) \
            / self.span * (rows - 1)
        column = (np.asarray(longitude, dtype='d') - self.west) \
            / self.span * (columns - 1)
        row = np.clip(row, 0.0, rows - 1.0)
        column = np.clip(column, 0.0, columns - 1.0)
        row0 = np.clip(np.floor(row).astype(np.intp), 0, rows - 2) \
            if rows > 1 else np.zeros_like(row, dtype=np.intp)
        col0 = np.clip(np.floor(column).astype(np.intp), 0, columns - 2) \
            if columns > 1 else np.zeros_like(column, dtype=np.intp)
        row1 = np.minimum(row0 + 1, rows - 1)
        col1 = np.minimum(col0 + 1, columns - 1)
        down = row - row0
        across = column - col0
        top = (self.samples[row0, col0] * (1.0 - across)
               + self.samples[row0, col1] * across)
        bottom = (self.samples[row1, col0] * (1.0 - across)
                  + self.samples[row1, col1] * across)
        return top * (1.0 - down) + bottom * down


def read_hgt(path: str) -> ElevationGrid:
    """Read an SRTM height file.

    Big-endian signed 16-bit samples with no header, a square grid whose side
    follows from the file's size, north row first (spec 1.1-1.3). Voids are
    filled from the ground around them (spec 1.5), because a hole read
    literally is a shaft thirty-two kilometres deep.
    """
    raw = np.fromfile(str(path), dtype='>i2')
    side = int(round(math.sqrt(raw.size)))
    if side < 2 or side * side != raw.size:
        raise ValueError(
            "%s holds %d samples, which is not a square grid; an SRTM height "
            "file is N by N with no header"
            % (os.path.basename(str(path)), raw.size))
    samples = raw.reshape(side, side).astype('d')
    south, west = corner_from_name(path)
    return ElevationGrid(samples=_fill_voids(samples), south=south, west=west)


def _fill_voids(samples: np.ndarray) -> np.ndarray:
    """Replace no-data samples with the mean of the ones that have data.

    Flat rather than clever: a void is where the radar saw nothing -- a lake, a
    steep shadow -- and a designer who cares what is really there will supply a
    filled product. What matters is that the hole is not a cliff.
    """
    void = samples <= VOID
    if not void.any():
        return samples
    filled = samples.copy()
    known = samples[~void]
    filled[void] = float(known.mean()) if known.size else 0.0
    return filled


@dataclass(frozen=True)
class LocalFrame:
    """Metres east and north of a point on the Earth, and back again.

    A tangent plane about the centre (spec 2.3), which holds over the few tens
    of kilometres anything a track editor frames covers. It is not a projection
    to use across a continent.
    """

    latitude: float
    longitude: float
    #: Metres per radian of latitude and of longitude at the centre (spec 2.2).
    per_latitude: float
    per_longitude: float

    def metres_from(self, latitude: Any, longitude: Any) -> tuple[Any, Any]:
        """``(east, north)`` in metres of a point, from the centre."""
        east = (np.radians(np.asarray(longitude, dtype='d') - self.longitude)
                * self.per_longitude)
        north = (np.radians(np.asarray(latitude, dtype='d') - self.latitude)
                 * self.per_latitude)
        return (east, north)

    def degrees_from(self, east: Any, north: Any) -> tuple[Any, Any]:
        """The ``(latitude, longitude)`` a point in metres stands on."""
        latitude = self.latitude + np.degrees(
            np.asarray(north, dtype='d') / self.per_latitude)
        longitude = self.longitude + np.degrees(
            np.asarray(east, dtype='d') / self.per_longitude)
        return (latitude, longitude)


def local_frame(latitude: float, longitude: float) -> LocalFrame:
    """The tangent plane at a point: how far a radian goes there (spec 2.2)."""
    phi = math.radians(float(latitude))
    limb = 1.0 - EARTH_ECCENTRICITY_SQUARED * math.sin(phi) ** 2
    meridional = (EARTH_SEMI_MAJOR * (1.0 - EARTH_ECCENTRICITY_SQUARED)
                  / limb ** 1.5)
    prime_vertical = EARTH_SEMI_MAJOR / math.sqrt(limb)
    return LocalFrame(latitude=float(latitude), longitude=float(longitude),
                      per_latitude=meridional,
                      per_longitude=prime_vertical * math.cos(phi))


@register_base
@dataclass(frozen=True)
class DEMBase(HeightBase):
    """A world's ground taken from an elevation file, about a point on the Earth.

    ``centre`` is the ``(latitude, longitude)`` the world's origin stands on,
    and is recorded in the project so a reopened track resolves to the same
    ground. ``datum`` is the height the world's origin is given: real ground is
    hundreds of metres above the sea, and a world whose waterline is at zero
    would have all of it underwater. ``relief`` scales what is left of it, for
    a valley whose real sides no road can climb.
    """

    KIND: ClassVar[str] = 'dem'
    #: The elevation file, as the project names it.
    path: str = ''
    centre: tuple[float, float] = (0.0, 0.0)
    #: Metres the world's origin sits at. None leaves the elevations alone.
    datum: float | None = None
    relief: float = 1.0
    _grid: list = field(default_factory=list, compare=False, repr=False)

    def grid(self) -> ElevationGrid:
        """The elevations, read once and kept.

        A bake samples the ground across every tile of a world; re-reading a
        twenty-five megabyte file for each of them is the whole of the bake.
        """
        if not self._grid:
            self._grid.append(read_hgt(self.path))
        grid: ElevationGrid = self._grid[0]
        return grid

    def frame(self) -> LocalFrame:
        """The tangent plane the world's metres are measured in."""
        return local_frame(self.centre[0], self.centre[1])

    def sample(self, x: Any, z: Any) -> np.ndarray:
        # The world's north is -z (spec 3.1), so a point further north is a
        # smaller z, and the sign here is the whole of that convention.
        latitude, longitude = self.frame().degrees_from(
            np.asarray(x, dtype='d'), -np.asarray(z, dtype='d'))
        ground = np.asarray(self.grid().at(latitude, longitude), dtype='d')
        if self.datum is not None:
            here = float(np.asarray(self.grid().at(*self.centre)))
            ground = ground - here + float(self.datum)
        if self.relief != 1.0:
            ground = ground * float(self.relief)
        return ground

    def to_json(self) -> dict[str, Any]:
        return {'kind': self.KIND, 'path': self.path,
                'centre': [float(self.centre[0]), float(self.centre[1])],
                'datum': self.datum, 'relief': float(self.relief)}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> DEMBase:
        centre = document.get('centre', (0.0, 0.0))
        datum = document.get('datum')
        return cls(path=str(document.get('path', '')),
                   centre=(float(centre[0]), float(centre[1])),
                   datum=None if datum is None else float(datum),
                   relief=float(document.get('relief', 1.0)))
