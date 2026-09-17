"""Shared helpers for the suite.

``surface_height`` answers the question every check about a baked world
eventually asks: at this (x, z), how high is the surface a triangle mesh
actually draws? Comparing a height *function* to a road proves nothing -- what
a viewer sees is the straight lines the mesh draws between its samples.
"""

import numpy as np
import pytest


def surface_height(positions, indices, x, z):
    """The highest surface a mesh presents at (x, z), or None if it has none.

    A vertical ray, tested against every triangle in barycentric coordinates.
    Skirts hang below the surface, so the ray can hit several triangles; the
    highest is the ground and the rest are curtain.
    """
    points = np.asarray(positions, 'd')
    tris = np.asarray(indices, np.int64).reshape(-1, 3)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    denominator = ((b[:, 2] - c[:, 2]) * (a[:, 0] - c[:, 0])
                   + (c[:, 0] - b[:, 0]) * (a[:, 2] - c[:, 2]))
    live = np.abs(denominator) > 1e-12
    u = np.zeros(len(tris))
    v = np.zeros(len(tris))
    safe = np.where(live, denominator, 1.0)
    u = ((b[:, 2] - c[:, 2]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (z - c[:, 2])) / safe
    v = ((c[:, 2] - a[:, 2]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (z - c[:, 2])) / safe
    w = 1.0 - u - v
    inside = live & (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
    if not inside.any():
        return None
    heights = u * a[:, 1] + v * b[:, 1] + w * c[:, 1]
    return float(heights[inside].max())


@pytest.fixture
def mesh_surface():
    """The helper above, as a fixture for tests that read better with one."""
    return surface_height


@pytest.fixture(autouse=True)
def asset_cache_is_not_the_real_one(tmp_path_factory, monkeypatch):
    """Keep the suite out of the user's own downloaded-asset cache.

    A run that wrote into it would leave the developer's cache full of test
    fixtures, and one that *read* from it would pass or fail depending on what
    that developer happened to have baked before.
    """
    monkeypatch.setenv('OPENGLCONTEXT_POLYHAVEN',
                       str(tmp_path_factory.mktemp('polyhaven')))
