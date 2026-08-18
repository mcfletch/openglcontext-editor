"""Placing things on the ground.

Scattering over a *height function* rather than over a meshed surface, which is
what a bake wants: the placements are decided once for the whole world, before
any tile exists, so the same tree lands in the same spot however the world is
partitioned and however many times it is re-baked.

:func:`scatter_on_heightfield` is the general form -- a density, a region, and
filters on where the ground is suitable. :func:`yaw_quaternions` turns the yaws
it produces into the quaternions an instance set carries.

Density is per square metre of *plan* area, not of surface area: a hillside
holds the same number of trees seen from above as the flat ground beside it,
which is how forestry measures it and how a designer thinks about it.

**Ask for a spacing rather than a density where you can.** Uniform random
candidates later thinned to a minimum separation is dart-throwing, and to
saturate the packing it has to be handed several times the number of instances
it will keep -- every one of which costs a height lookup, four more for the
slope, and whatever the caller's own filter costs. A jittered grid at the
spacing asks for the answer directly.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from OpenGLContext.loaders.tiles3d.scatter import Scatter

from OpenGLContext_editor.bake.bounds import BoundingBox

HeightFn = Callable[[Any, Any], Any]

#: How far apart the samples are that measure the ground's slope, in metres. Far
#: enough that a metre-scale wrinkle does not read as a cliff, close enough to
#: catch a real bank.
SLOPE_STEP = 1.0

#: How far a candidate may wander from its cell's centre, as a fraction of the
#: cell. Under 1, so no two candidates can meet: what is wanted is a set that
#: is not a lattice, not one that has to be thinned again afterwards.
GRID_JITTER = 0.86


def scatter_on_heightfield(
        height_fn: HeightFn, extent: BoundingBox, density: float | None = None,
        seed: int = 0, scale_range: tuple[float, float] = (1.0, 1.0),
        slope_limit: float | None = None,
        height_range: tuple[float, float] | None = None,
        keep: Callable[[np.ndarray], np.ndarray] | None = None,
        spacing: float | None = None,
        slope_fn: HeightFn | None = None) -> Scatter:
    """Place instances over a region of ground.

    Say either how many or how far apart. ``density`` is instances per square
    metre of the region's footprint, placed at random; ``spacing`` is metres
    between them, placed one to a cell of a jittered grid. Prefer ``spacing``
    for anything that will be thinned to a minimum separation afterwards: a
    random set has to be several times too dense before that thinning saturates,
    and every candidate in it is paid for.

    ``slope_limit`` drops anything on ground steeper than that many degrees;
    ``height_range`` keeps only placements between two elevations -- above the
    waterline and below the treeline, say. ``keep`` is a free filter over the
    (N,3) candidate positions for anything those two do not express, such as a
    distance-to-road mask.

    ``slope_fn(x, z) -> radians`` is how steep the ground is, for a caller that
    has a cheaper way to say than four more height lookups apiece -- a bake
    samples the ground into a height field before it scatters anything on it,
    and that field can answer. It answers *better*, too, at the resolution a
    tree cares about: a central difference over a metre reads every wrinkle of
    an earthwork as a cliff.

    The filters run cheapest first, each on what the last one left: the
    elevation band is free once the height is known, ``keep`` is the caller's
    own cost, and the slope is four more height lookups apiece.

    The result is the engine's own :class:`~OpenGLContext.loaders.tiles3d.scatter.Scatter`
    -- positions, yaws and uniform scales -- so it feeds the same instanced
    drawing as a tile-time scatter.
    """
    rng = np.random.default_rng(seed)
    if spacing is not None:
        x, z = _grid_candidates(extent, spacing, rng)
    elif density is not None:
        x, z = _random_candidates(extent, density, rng)
    else:
        raise ValueError(
            "a scatter needs either a density or a spacing to place anything")
    if not len(x):
        return Scatter(np.zeros((0, 3), 'f4'), np.zeros(0), np.zeros(0))
    yaws = rng.random(len(x)) * (2.0 * math.pi)
    scales = rng.uniform(scale_range[0], scale_range[1], size=len(x))

    y = np.asarray(height_fn(x, z), dtype='d')
    live = np.ones(len(x), dtype=bool)
    if height_range is not None:
        live &= (y >= height_range[0]) & (y <= height_range[1])
    x, z, y, yaws, scales = (part[live] for part in (x, z, y, yaws, scales))
    positions = np.stack([x, y, z], axis=-1)
    if keep is not None and len(x):
        live = np.asarray(keep(positions), dtype=bool)
        x, z, yaws, scales = (part[live] for part in (x, z, yaws, scales))
        positions = positions[live]
    if slope_limit is not None and len(x):
        measured = (surface_slope(height_fn, x, z) if slope_fn is None
                    else np.asarray(slope_fn(x, z), dtype='d'))
        live = measured <= math.radians(slope_limit)
        positions, yaws, scales = positions[live], yaws[live], scales[live]
    return Scatter(positions.astype('f4'), yaws, scales)


def _random_candidates(extent: BoundingBox, density: float,
                       rng: Any) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random points over the region, ``density`` to the square metre."""
    footprint = float(extent.size[0]) * float(extent.size[2])
    count = max(int(round(footprint * density)), 0)
    return (rng.uniform(extent.minimum[0], extent.maximum[0], size=count),
            rng.uniform(extent.minimum[2], extent.maximum[2], size=count))


def _grid_candidates(extent: BoundingBox, spacing: float,
                     rng: Any) -> tuple[np.ndarray, np.ndarray]:
    """One point per cell of a jittered grid of ``spacing`` metres."""
    if spacing <= 0.0:
        raise ValueError("a scatter's spacing is a distance in metres, not %r"
                         % (spacing,))
    axis_x = np.arange(float(extent.minimum[0]), float(extent.maximum[0])
                       + spacing * 0.5, spacing)
    axis_z = np.arange(float(extent.minimum[2]), float(extent.maximum[2])
                       + spacing * 0.5, spacing)
    grid_x, grid_z = np.meshgrid(axis_x, axis_z)
    count = grid_x.size
    wander = spacing * GRID_JITTER / 2.0
    return (grid_x.ravel() + rng.uniform(-wander, wander, size=count),
            grid_z.ravel() + rng.uniform(-wander, wander, size=count))


def surface_slope(height_fn: HeightFn, x: Any, z: Any,
                  step: float = SLOPE_STEP) -> np.ndarray:
    """The ground's angle from horizontal at each (x, z), in radians.

    Central differences over ``step`` metres, so the answer describes the slope
    a tree would stand on rather than the roughness of the mesh under it.
    """
    x = np.asarray(x, dtype='d')
    z = np.asarray(z, dtype='d')
    dx = (np.asarray(height_fn(x + step, z), 'd')
          - np.asarray(height_fn(x - step, z), 'd')) / (2.0 * step)
    dz = (np.asarray(height_fn(x, z + step), 'd')
          - np.asarray(height_fn(x, z - step), 'd')) / (2.0 * step)
    angle: np.ndarray = np.arctan(np.sqrt(dx * dx + dz * dz))
    return angle


def yaw_quaternions(yaws: Any) -> np.ndarray:
    """Yaws in radians as (N,4) xyzw quaternions about +Y."""
    angles = np.asarray(yaws, dtype='d') * 0.5
    out = np.zeros((len(angles), 4), dtype='f')
    out[:, 1] = np.sin(angles)
    out[:, 3] = np.cos(angles)
    return out


def settle_onto(height_fn: HeightFn, positions: Any,
                offset: float = 0.0) -> np.ndarray:
    """Move points vertically onto the ground, plus an offset.

    A placement authored on a map has no height; this gives it one. The offset
    sinks a tree's root below the surface, or lifts a sign above it.
    """
    points = np.array(positions, dtype='d').reshape(-1, 3)
    points[:, 1] = np.asarray(height_fn(points[:, 0], points[:, 2]), 'd') + offset
    return points
