"""Every metre of the road is written, and written once.

A tileset refines by *replacement*: once a tile's children are drawn, the tile
itself is not. So a stretch of road that a parent tile carries and its children
do not is a stretch that vanishes the moment the viewer gets close enough to
refine -- the carriageway runs into the ground and stops. The other way round is
as bad: two tiles writing the same stretch put two coplanar surfaces in the
depth buffer and the join flickers across the road as the camera moves.

So at every level of the tree, the tiles at that level between them write each
segment of the centreline exactly once.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.road import RoadLayer, RoadPath

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
SIDE = 1024.0
WORLD = BoundingBox((-SIDE / 2, -200.0, -SIDE / 2), (SIDE / 2, 200.0, SIDE / 2))


def _oval(count=241, across=380.0, along=300.0):
    angle = np.linspace(0.0, 2.0 * np.pi, count)
    return np.stack([np.cos(angle) * across,
                     np.sin(angle * 3.0) * 8.0,
                     np.sin(angle) * along], axis=-1)


def _quadrants(region, depth):
    """The regions of one level of a quadtree over ``region``."""
    if depth <= 0:
        return [region]
    out = []
    low, high = np.asarray(region.minimum), np.asarray(region.maximum)
    middle = (low + high) / 2.0
    for x in (0, 1):
        for z in (0, 1):
            out.extend(_quadrants(BoundingBox(
                (low[0] if not x else middle[0], low[1],
                 low[2] if not z else middle[2]),
                (middle[0] if not x else high[0], high[1],
                 middle[2] if not z else high[2])), depth - 1))
    return out


def _written(layer, regions, error):
    """How many of the regions write each segment of the centreline."""
    line = layer.path.resampled(layer.spacing_for(error))
    counted = np.zeros(max(len(line) - 1, 1))
    for region in regions:
        for index in layer.segments_in(region, error):
            counted[index] += 1
    return counted


@pytest.fixture(scope='module')
def layer():
    return RoadLayer(RoadPath(_oval(), profile=PROFILE))


class TestAndTheGeometryFollowsIt:
    def test_a_tile_that_writes_no_segment_writes_no_road(self, layer) -> None:
        away = BoundingBox((2000.0, -200.0, 2000.0), (3000.0, 200.0, 3000.0))
        assert layer.segments_in(away, 1.0) == set()
        assert layer.content(away, 1.0) == []

    def test_a_tile_that_writes_some_writes_a_surface(self, layer) -> None:
        assert layer.segments_in(WORLD, 1.0)
        assert [node for node in layer.content(WORLD, 1.0)
                if node.name == layer.name]

    def test_the_surface_is_as_long_as_the_responsibility(self, layer) -> None:
        """One ring of vertices per point, so the count says which points."""
        region = _quadrants(WORLD, 1)[0]
        rings = sum(len(np.unique(np.round(node.mesh.positions[:, [0, 2]], 4),
                                  axis=0))
                    for node in layer.content(region, 1.0)
                    if node.name == layer.name)
        assert rings > len(layer.segments_in(region, 1.0))


class TestOneLevelOfTheTree:
    @pytest.mark.parametrize('depth', [0, 1, 2, 3])
    def test_every_segment_is_written_somewhere(self, layer, depth) -> None:
        error = 3.0 / max(2 ** depth, 1)
        counted = _written(layer, _quadrants(WORLD, depth), error)
        assert int((counted == 0).sum()) == 0

    @pytest.mark.parametrize('depth', [1, 2, 3])
    def test_no_segment_is_written_twice(self, layer, depth) -> None:
        """Not a cosmetic point: two coplanar road surfaces fight for the
        depth buffer and the join shows as a band flickering across it."""
        error = 3.0 / max(2 ** depth, 1)
        counted = _written(layer, _quadrants(WORLD, depth), error)
        assert int((counted > 1).sum()) == 0


class TestTheShippedCircuit:
    def test_no_stretch_of_it_goes_unwritten(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=1024.0, field_resolution=257,
                                control_size=256)
        found = world.circuit_layer()
        half = world.extent / 2.0
        region = BoundingBox((-half, -400.0, -half), (half, 400.0, half))
        counted = _written(found, _quadrants(region, 2), 0.75)
        assert int((counted == 0).sum()) == 0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
