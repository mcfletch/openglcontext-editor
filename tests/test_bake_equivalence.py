"""The baker against the engine's own terrain baker: equivalent, or better.

The engine already bakes a terrain quadtree
(:func:`OpenGLContext.loaders.tiles3d.procedural.build_terrain_tileset`). It is
a *reference*, not an oracle: this baker is free to order vertices differently,
carry attributes the other omits, and refine further. What it may not do is
represent the ground worse.

So both are pointed at the same height function and measured against that
function -- how far the baked surface is from the surface it was made from, at
points that are not vertices of either. Better passes; only worse fails.
"""

import json
import os

import numpy as np
import pytest
from OpenGLContext.loaders import gltf
from OpenGLContext.loaders.tiles3d.procedural import (
    WATER_LEVEL,
    build_terrain_tileset,
    terrain_colors,
    terrain_height,
)
from OpenGLContext.loaders.tiles3d.tileset import build_runtime_tileset

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.driver import bake_world
from OpenGLContext_editor.bake.layers import HeightfieldLayer

EXTENT = 2048.0
RESOLUTION = 33
LEVELS = 3                  # the reference's levels; depth LEVELS-1 for the baker
SAMPLES = 400


def _truth(x, z):
    """The surface both bakers are approximating, water clamped as they clamp it."""
    return np.maximum(terrain_height(x, z), WATER_LEVEL)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp('reference')
    path = build_terrain_tileset(str(directory), extent=EXTENT, levels=LEVELS,
                                 tile_res=RESOLUTION, height_fn=terrain_height)
    return _Surface(path)


@pytest.fixture(scope='module')
def baked(tmp_path_factory):
    directory = tmp_path_factory.mktemp('baked')
    half = EXTENT / 2.0
    layer = HeightfieldLayer(
        height_fn=terrain_height,
        extent=BoundingBox((-half, 0, -half), (half, 0, half)),
        resolution=RESOLUTION, color_fn=terrain_colors, water_level=WATER_LEVEL)
    result = bake_world([layer], str(directory), depth=LEVELS - 1)
    return _Surface(result.tileset)


class _Surface:
    """A baked tileset, loaded so its finest ground can be sampled."""

    def __init__(self, tileset_path):
        self.path = tileset_path
        with open(tileset_path) as handle:
            document = json.load(handle)
        base = os.path.dirname(os.path.abspath(tileset_path)) + os.sep
        self.tileset = build_runtime_tileset(document, base_uri=base, recenter=True)
        self.leaves = [tile for tile in self.tileset.root.iter_tiles()
                       if not tile.children and tile.content_uri]
        self._meshes = {}

    def tiles(self):
        return list(self.tileset.root.iter_tiles())

    def _mesh(self, uri):
        if uri not in self._meshes:
            self._meshes[uri] = _triangles(uri)
        return self._meshes[uri]

    def height_at(self, x, z):
        """The baked surface's height at (x, z), or None where it has no ground."""
        for tile in self.leaves:
            box = tile.bounding_volume
            centre, radius = box.bounding_sphere()
            if abs(x - centre[0]) > radius or abs(z - centre[2]) > radius:
                continue
            points, tris = self._mesh(tile.content_uri)
            height = _sample(points, tris, x, z)
            if height is not None:
                return height
        return None

    def footprint(self):
        low = np.array([np.inf, np.inf])
        high = -low
        for tile in self.leaves:
            points, _ = self._mesh(tile.content_uri)
            low = np.minimum(low, points[:, [0, 2]].min(axis=0))
            high = np.maximum(high, points[:, [0, 2]].max(axis=0))
        return low, high


def _triangles(uri):
    """A tile's positions and triangle indices, in world coordinates."""
    scene = gltf.load_gltf(uri)
    points, tris = [], []
    stack = [scene.group]
    while stack:
        node = stack.pop()
        geometry = getattr(node, 'geometry', None)
        if geometry is not None and getattr(geometry, 'positions', None) is not None:
            offset = len(points and np.vstack(points) or [])
            indices = geometry.indices
            if indices is None:
                indices = np.arange(len(geometry.positions), dtype=np.uint32)
            points.append(np.asarray(geometry.positions, 'd'))
            tris.append(np.asarray(indices, np.int64).reshape(-1, 3) + offset)
        stack.extend(getattr(node, 'children', None) or [])
    return np.vstack(points), np.vstack(tris)


def _sample(points, tris, x, z):
    """The surface height at (x, z) by vertical ray, or None if it misses.

    The skirt hangs vertically, so a ray can pass through several triangles; the
    highest hit is the ground and the rest are curtain.
    """
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    ax, az = a[:, 0], a[:, 2]
    bx, bz = b[:, 0], b[:, 2]
    cx, cz = c[:, 0], c[:, 2]
    denominator = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    live = np.abs(denominator) > 1e-12
    u = np.zeros(len(tris))
    v = np.zeros(len(tris))
    u[live] = (((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / denominator)[live]
    v[live] = (((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / denominator)[live]
    w = 1.0 - u - v
    inside = live & (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
    if not inside.any():
        return None
    heights = u * a[:, 1] + v * b[:, 1] + w * c[:, 1]
    return float(heights[inside].max())


def _sample_points(count=SAMPLES, seed=5):
    """Points inside the world, deliberately off both bakers' vertex grids."""
    rng = np.random.default_rng(seed)
    half = EXTENT / 2.0 - 8.0
    return rng.uniform(-half, half, size=(count, 2))


def _errors(surface, points):
    got, expected = [], []
    for x, z in points:
        height = surface.height_at(float(x), float(z))
        if height is None:
            continue
        got.append(height)
        expected.append(float(_truth(np.array([x]), np.array([z]))[0]))
    return np.abs(np.array(got) - np.array(expected)), len(got)


class TestTheSurfaceIsAtLeastAsFaithful:
    def test_both_bakers_cover_the_same_points(self, reference, baked) -> None:
        points = _sample_points()
        _, reference_hits = _errors(reference, points)
        _, baked_hits = _errors(baked, points)
        assert baked_hits >= reference_hits
        assert baked_hits >= len(points) * 0.99

    def test_the_mean_error_is_no_worse(self, reference, baked) -> None:
        points = _sample_points()
        reference_error, _ = _errors(reference, points)
        baked_error, _ = _errors(baked, points)
        assert baked_error.mean() <= reference_error.mean() * 1.05

    def test_the_worst_error_is_no_worse(self, reference, baked) -> None:
        points = _sample_points()
        reference_error, _ = _errors(reference, points)
        baked_error, _ = _errors(baked, points)
        assert baked_error.max() <= reference_error.max() * 1.05

    def test_the_error_is_small_against_the_terrain_s_own_relief(
            self, reference, baked) -> None:
        """Not merely equal to the reference -- close to the real surface."""
        baked_error, _ = _errors(baked, _sample_points())
        assert baked_error.mean() < 2.0        # metres, over 400 m of relief


class TestTheWorldIsTheSameWorld:
    def test_the_footprints_match(self, reference, baked) -> None:
        reference_low, reference_high = reference.footprint()
        baked_low, baked_high = baked.footprint()
        assert np.allclose(baked_low, reference_low, atol=EXTENT * 0.01)
        assert np.allclose(baked_high, reference_high, atol=EXTENT * 0.01)

    def test_it_refines_at_least_as_far(self, reference, baked) -> None:
        assert len(baked.leaves) >= len(reference.leaves)

    def test_the_error_ladder_is_monotone(self, baked) -> None:
        for tile in baked.tiles():
            for child in tile.children:
                assert child.geometric_error <= tile.geometric_error

    def test_no_tile_is_missing_from_the_middle_of_the_world(self, baked) -> None:
        """A hole would show as a sample point with ground in the reference and
        none here; the coverage test above catches that, and this states the
        count the tree should have."""
        assert len(baked.leaves) == 4 ** (LEVELS - 1)


class TestWhatTheBakerAddsOnTop:
    def test_it_carries_vertex_colour_as_the_reference_does(self, baked) -> None:
        scene = gltf.load_gltf(baked.leaves[0].content_uri)
        mesh = _first_mesh(scene.group)
        assert mesh.colors is not None

    def test_it_carries_normals_the_reference_also_carries(self, baked) -> None:
        mesh = _first_mesh(gltf.load_gltf(baked.leaves[0].content_uri).group)
        assert mesh.normals is not None
        assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-3)

    def test_its_tiles_are_no_larger_than_the_reference_s(self, reference,
                                                          baked) -> None:
        """Same ground, same vertex count: the file should not be bigger."""
        def biggest(surface):
            return max(os.path.getsize(tile.content_uri) for tile in surface.leaves)
        assert biggest(baked) <= biggest(reference) * 1.1


def _first_mesh(node):
    stack = [node]
    while stack:
        current = stack.pop()
        geometry = getattr(current, 'geometry', None)
        if geometry is not None and getattr(geometry, 'positions', None) is not None:
            return geometry
        stack.extend(getattr(current, 'children', None) or [])
    raise AssertionError('no mesh in the tile')
