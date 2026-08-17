"""The road carries the shade of the wood it runs through.

The forest is cleared out of the corridor, so what falls across the tarmac is
what the trees either side lean over -- and where the alignment turns to face
the sun, or the canopy opens, the road is lit. Baked into the surface's vertex
colours, because the trees do not move and neither does the sun.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.road import RoadLayer, RoadPath

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
REGION = BoundingBox((-400.0, -50.0, -50.0), (400.0, 100.0, 450.0))


def _path(count=41, length=400.0):
    z = np.linspace(0.0, length, count)
    return RoadPath(np.stack([np.zeros(count), np.zeros(count), z], axis=-1),
                    profile=PROFILE)


def _dark_north(x, z):
    """Deep shade past the halfway point, full sun before it."""
    return np.where(np.asarray(z, 'd') > 200.0, 0.2, 1.0)


def _meshes(**named):
    return [node.mesh for node
            in RoadLayer(_path(), **named).content(REGION, error=1.0)
            if node.name == 'road']


class TestWhatTheSurfaceCarries:
    def test_a_road_told_nothing_is_left_at_full_sun(self) -> None:
        assert all(mesh.colors is None for mesh in _meshes())

    def test_a_shaded_road_carries_it_per_vertex(self) -> None:
        for mesh in _meshes(shade=_dark_north):
            assert len(mesh.colors) == len(mesh.positions)

    def test_the_shaded_stretch_is_darker(self) -> None:
        for mesh in _meshes(shade=_dark_north):
            north = mesh.positions[:, 2] > 250.0
            south = mesh.positions[:, 2] < 150.0
            if north.any() and south.any():
                assert float(mesh.colors[north, 0].mean()) < 0.3
                assert float(mesh.colors[south, 0].mean()) > 0.9
                return
        raise AssertionError("no mesh spanned both stretches")

    def test_a_coarse_tile_is_shaded_the_same_way(self) -> None:
        """The road's level of detail is its point spacing, and the shade is
        asked for the points that are actually written."""
        meshes = [node.mesh for node
                  in RoadLayer(_path(), shade=_dark_north).content(REGION,
                                                                   error=40.0)
                  if node.name == 'road']
        assert meshes and all(len(m.colors) == len(m.positions) for m in meshes)


class TestTheShippedWorld:
    def test_its_circuit_runs_through_its_own_shade(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        from OpenGLContext_editor.world.species import species_are_available
        if not species_are_available():
            pytest.skip("the example world's trees are not installed")
        world = ProceduralWorld(extent=1024.0, field_resolution=257,
                                control_size=256)
        shade = world.canopy_shade()
        assert shade is not None
        line = world.circuit().points
        found = np.asarray(shade(line[:, 0], line[:, 2]))
        assert float(found.min()) < 0.85
        assert float(found.max()) > 0.5

    def test_a_world_with_no_forest_leaves_it_at_full_sun(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        assert ProceduralWorld(forest='tiles', ground='tiles',
                               extent=512.0).canopy_shade() is None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
