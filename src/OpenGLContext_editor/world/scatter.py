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


def scatter_on_heightfield(
        height_fn: HeightFn, extent: BoundingBox, density: float, seed: int,
        scale_range: tuple[float, float] = (1.0, 1.0),
        slope_limit: float | None = None,
        height_range: tuple[float, float] | None = None,
        keep: Callable[[np.ndarray], np.ndarray] | None = None) -> Scatter:
    """Place instances over a region of ground.

    ``density`` is instances per square metre of the region's footprint, before
    filtering. ``slope_limit`` drops anything on ground steeper than that many
    degrees; ``height_range`` keeps only placements between two elevations --
    above the waterline and below the treeline, say. ``keep`` is a free filter
    over the (N,3) candidate positions for anything those two do not express,
    such as a distance-to-road mask.

    The result is the engine's own :class:`~OpenGLContext.loaders.tiles3d.scatter.Scatter`
    -- positions, yaws and uniform scales -- so it feeds the same instanced
    drawing as a tile-time scatter.
    """
    footprint = float(extent.size[0]) * float(extent.size[2])
    count = int(round(footprint * density))
    if count <= 0:
        return Scatter(np.zeros((0, 3), 'f4'), np.zeros(0), np.zeros(0))

    rng = np.random.default_rng(seed)
    x = rng.uniform(extent.minimum[0], extent.maximum[0], size=count)
    z = rng.uniform(extent.minimum[2], extent.maximum[2], size=count)
    y = np.asarray(height_fn(x, z), dtype='d')
    positions = np.stack([x, y, z], axis=-1)
    yaws = rng.random(count) * (2.0 * math.pi)
    scales = rng.uniform(scale_range[0], scale_range[1], size=count)

    mask = np.ones(count, dtype=bool)
    if height_range is not None:
        mask &= (y >= height_range[0]) & (y <= height_range[1])
    if slope_limit is not None:
        mask &= surface_slope(height_fn, x, z) <= math.radians(slope_limit)
    if keep is not None:
        mask &= np.asarray(keep(positions), dtype=bool)
    return Scatter(positions[mask].astype('f4'), yaws[mask], scales[mask])


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
