"""What a tile's content is made of: a sampled heightfield, and placed instances.

A layer answers one question -- "what is in this region, at this error?" -- and
these assert the two answers the first world needs: terrain meshed to the
region's footprint, and instances that fall inside it thinned to suit the tile.
"""

import numpy as np
import pytest
from OpenGLContext.loaders.gltf.writer import SceneNode
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.layers import HeightfieldLayer, InstanceLayer, MeshLayer


def _slope(x, z):
    """A height field with a known value everywhere: y = x/10 + z/20."""
    return np.asarray(x) / 10.0 + np.asarray(z) / 20.0


def _footprint(low_x, high_x, low_z, high_z):
    return BoundingBox((low_x, -1000, low_z), (high_x, 1000, high_z))


def _positions(nodes):
    return np.vstack([node.mesh.positions for node in nodes])


class TestAHeightfieldLayer:
    def _layer(self, **kwargs):
        return HeightfieldLayer(height_fn=_slope,
                                extent=_footprint(-100, 100, -100, 100),
                                resolution=9, **kwargs)

    def test_its_bounds_follow_the_surface(self) -> None:
        box = self._layer().bounds()
        assert box.minimum[0] == -100 and box.maximum[0] == 100
        assert box.minimum[1] == pytest.approx(-15.0, abs=0.5)   # x/10 + z/20 at -100
        assert box.maximum[1] == pytest.approx(15.0, abs=0.5)

    def test_it_meshes_the_region_it_is_asked_for(self) -> None:
        nodes = self._layer().content(_footprint(0, 50, 0, 50), error=8.0)
        positions = _positions(nodes)
        assert positions[:, 0].min() >= -0.001 and positions[:, 0].max() <= 50.001
        assert positions[:, 2].min() >= -0.001 and positions[:, 2].max() <= 50.001

    def test_the_surface_it_meshes_is_the_height_function(self) -> None:
        nodes = self._layer().content(_footprint(0, 50, 0, 50), error=8.0)
        mesh = nodes[0].mesh
        # Skirt vertices hang below; the surface ones sit on the function.
        surface = mesh.positions[:81]
        expected = _slope(surface[:, 0], surface[:, 2])
        assert np.allclose(surface[:, 1], expected, atol=1e-4)

    def test_it_declines_a_region_outside_its_extent(self) -> None:
        assert self._layer().content(_footprint(500, 600, 0, 50), error=8.0) == []

    def test_it_declines_a_region_the_surface_misses_vertically(self) -> None:
        """An octant well above the terrain holds no terrain."""
        above = BoundingBox((0, 500, 0), (50, 600, 50))
        assert self._layer().content(above, error=8.0) == []

    def test_it_clips_a_region_to_its_extent(self) -> None:
        nodes = self._layer().content(_footprint(50, 300, 0, 50), error=8.0)
        assert _positions(nodes)[:, 0].max() <= 100.001

    def test_the_vertex_count_is_the_resolution_it_was_given(self) -> None:
        layer = self._layer(skirt=0.0)
        mesh = layer.content(_footprint(0, 50, 0, 50), error=8.0)[0].mesh
        assert len(mesh.positions) == 9 * 9

    def test_a_skirt_hangs_below_the_surface(self) -> None:
        with_skirt = self._layer(skirt=3.0).content(_footprint(0, 50, 0, 50), 8.0)[0]
        without = self._layer(skirt=0.0).content(_footprint(0, 50, 0, 50), 8.0)[0]
        assert len(with_skirt.mesh.positions) > len(without.mesh.positions)
        assert with_skirt.mesh.positions[:, 1].min() < without.mesh.positions[:, 1].min()

    def test_it_colours_what_it_meshes(self) -> None:
        def blue(positions, normals):
            return np.tile(np.array([0, 0, 1], 'f'), (len(positions), 1))
        layer = self._layer(color_fn=blue)
        mesh = layer.content(_footprint(0, 50, 0, 50), error=8.0)[0].mesh
        assert np.allclose(mesh.colors[:, :3], (0, 0, 1))

    def test_water_can_be_clamped_flat(self) -> None:
        layer = self._layer(water_level=5.0, skirt=0.0)
        mesh = layer.content(_footprint(-100, 0, -100, 0), error=8.0)[0].mesh
        assert mesh.positions[:, 1].min() >= 5.0 - 1e-5

    def test_it_carries_the_material_it_was_given(self) -> None:
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        material = PBRMaterial(baseColor=(0.2, 0.3, 0.4))
        layer = self._layer(material=material)
        mesh = layer.content(_footprint(0, 50, 0, 50), error=8.0)[0].mesh
        assert mesh.material is material


def _prototype(name='tree'):
    return PBRMesh(positions=np.array([(0, 0, 0), (1, 0, 0), (0, 2, 0)], 'f'),
                   indices=np.array([0, 1, 2], np.uint32))


class TestAnInstanceLayer:
    def _layer(self, count=64, **kwargs):
        rng = np.random.default_rng(7)
        positions = rng.uniform(-100, 100, size=(count, 3))
        positions[:, 1] = 0.0
        return InstanceLayer(positions=positions, lods=[(0.0, _prototype())],
                             **kwargs), positions

    def test_its_bounds_enclose_its_placements(self) -> None:
        layer, positions = self._layer()
        box = layer.bounds()
        assert all(box.contains(p) for p in positions)

    def test_it_places_the_instances_inside_the_region(self) -> None:
        layer, positions = self._layer()
        region = BoundingBox((-100, -10, -100), (0, 10, 0))
        nodes = layer.content(region, error=1.0)
        expected = sum(1 for p in positions if region.contains(p))
        assert nodes[0].instances.count() == expected

    def test_an_empty_region_yields_nothing(self) -> None:
        layer, _ = self._layer()
        assert layer.content(BoundingBox((500, 0, 500), (600, 1, 600)), 1.0) == []

    def test_rotations_and_scales_ride_along(self) -> None:
        rng = np.random.default_rng(3)
        positions = rng.uniform(-10, 10, size=(8, 3))
        rotations = np.tile(np.array([0, 0, 0, 1.0]), (8, 1))
        scales = rng.uniform(0.5, 2.0, size=(8, 3))
        layer = InstanceLayer(positions=positions, rotations=rotations,
                              scales=scales, lods=[(0.0, _prototype())])
        instances = layer.content(BoundingBox((-20, -20, -20), (20, 20, 20)), 1.0)
        assert instances[0].instances.rotations.shape == (8, 4)
        assert instances[0].instances.scales.shape == (8, 3)

    def test_a_uniform_scale_expands_to_three_axes(self) -> None:
        positions = np.zeros((4, 3))
        layer = InstanceLayer(positions=positions, scales=np.array([1.0, 2, 3, 4]),
                              lods=[(0.0, _prototype())])
        instances = layer.content(BoundingBox((-1, -1, -1), (1, 1, 1)), 1.0)
        assert instances[0].instances.scales.shape == (4, 3)
        assert np.allclose(instances[0].instances.scales[3], (4, 4, 4))

    def test_a_coarse_tile_is_thinned_to_the_instance_budget(self) -> None:
        layer, _ = self._layer(count=200, max_instances=32)
        region = BoundingBox((-200, -50, -200), (200, 50, 200))
        assert layer.content(region, error=64.0)[0].instances.count() <= 32

    def test_thinning_is_deterministic(self) -> None:
        layer, _ = self._layer(count=200, max_instances=32)
        region = BoundingBox((-200, -50, -200), (200, 50, 200))
        first = layer.content(region, 64.0)[0].instances.translations
        second = layer.content(region, 64.0)[0].instances.translations
        assert np.array_equal(first, second)

    def test_a_fine_tile_keeps_every_instance(self) -> None:
        layer, positions = self._layer(count=40, max_instances=1024)
        region = BoundingBox((-200, -50, -200), (200, 50, 200))
        assert layer.content(region, error=1.0)[0].instances.count() == 40

    def test_the_ladder_picks_a_coarser_mesh_for_a_coarser_tile(self) -> None:
        near, far = _prototype('near'), _prototype('far')
        rng = np.random.default_rng(1)
        layer = InstanceLayer(positions=rng.uniform(-10, 10, size=(4, 3)),
                              lods=[(0.0, near), (20.0, far)])
        region = BoundingBox((-20, -20, -20), (20, 20, 20))
        assert layer.content(region, error=1.0)[0].mesh is near
        assert layer.content(region, error=50.0)[0].mesh is far

    def test_an_empty_ladder_is_refused(self) -> None:
        with pytest.raises(ValueError, match='at least one'):
            InstanceLayer(positions=np.zeros((1, 3)), lods=[])


class TestAMeshLayer:
    def test_it_emits_meshes_that_meet_the_region(self) -> None:
        inside = SceneNode(mesh=_prototype(), translation=(0, 0, 0))
        outside = SceneNode(mesh=_prototype(), translation=(500, 0, 500))
        layer = MeshLayer(nodes=[inside, outside])
        got = layer.content(BoundingBox((-10, -10, -10), (10, 10, 10)), error=1.0)
        assert got == [inside]

    def test_its_bounds_cover_every_node(self) -> None:
        layer = MeshLayer(nodes=[SceneNode(mesh=_prototype(), translation=(50, 0, 0))])
        assert layer.bounds().maximum[0] == pytest.approx(51.0)

    def test_it_can_be_held_back_from_coarse_tiles(self) -> None:
        layer = MeshLayer(nodes=[SceneNode(mesh=_prototype())], maximum_error=10.0)
        region = BoundingBox((-10, -10, -10), (10, 10, 10))
        assert layer.content(region, error=50.0) == []
        assert layer.content(region, error=5.0) != []


class TestMeasuringAPlacedNode:
    def test_a_rotated_node_is_measured_at_its_widest(self) -> None:
        """A box that a turned mesh sticks out of gets the tile culled while it
        is still on screen, so rotation widens the box to enclose every turn."""
        from OpenGLContext_editor.bake.layers import node_bounds
        upright = node_bounds(SceneNode(mesh=_prototype()))
        turned = node_bounds(SceneNode(mesh=_prototype(),
                                       rotation=(0, 0.7071, 0, 0.7071)))
        assert np.all(turned.size >= upright.size - 1e-9)
        assert turned.size[0] > upright.size[0]

    def test_a_scaled_node_is_measured_scaled(self) -> None:
        from OpenGLContext_editor.bake.layers import node_bounds
        box = node_bounds(SceneNode(mesh=_prototype(), scale=(2, 2, 2)))
        assert box.maximum[1] == pytest.approx(4.0)      # the 2-unit-tall triangle

    def test_a_negative_scale_does_not_invert_the_box(self) -> None:
        from OpenGLContext_editor.bake.layers import node_bounds
        box = node_bounds(SceneNode(mesh=_prototype(), scale=(-1, 1, 1)))
        assert np.all(box.minimum <= box.maximum)

    def test_a_node_with_no_mesh_has_no_bounds(self) -> None:
        from OpenGLContext_editor.bake.layers import node_bounds
        assert node_bounds(SceneNode(name='empty')) is None

    def test_several_primitives_are_measured_together(self) -> None:
        from OpenGLContext_editor.bake.layers import node_bounds
        tall = PBRMesh(positions=np.array([(0, 0, 0), (0, 9, 0), (1, 0, 0)], 'f'))
        box = node_bounds(SceneNode(mesh=[_prototype(), tall]))
        assert box.maximum[1] == pytest.approx(9.0)

    def test_rotated_instances_are_measured_at_their_widest(self) -> None:
        from OpenGLContext.loaders.gltf.writer import InstanceSet

        from OpenGLContext_editor.bake.layers import node_bounds
        instances = InstanceSet(translations=np.zeros((2, 3), 'f'),
                                rotations=np.tile([0, 0, 0, 1.0], (2, 1)),
                                scales=np.array([(1, 1, 1), (3, 3, 3)], 'f'))
        box = node_bounds(SceneNode(mesh=_prototype(), instances=instances))
        assert box.maximum[0] > 3.0     # the widened radius, three times over


class TestALadderRungOfSeveralMeshes:
    """A tree is a bark trunk and alpha-masked needles: two materials, and so
    two meshes. A rung of the ladder is whatever the prototype is made of."""

    def _placed(self, rung):
        from OpenGLContext_editor.bake.layers import InstanceLayer
        return InstanceLayer(positions=np.array([(0.0, 0.0, 0.0),
                                                 (4.0, 0.0, 4.0)]),
                             lods=[(0.0, rung)], name='trees')

    def _region(self):
        from OpenGLContext_editor.bake.bounds import BoundingBox
        return BoundingBox((-50.0, -50.0, -50.0), (50.0, 50.0, 50.0))

    def _mesh(self, colour):
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        from OpenGLContext.scenegraph.pbrmesh import PBRMesh
        return PBRMesh(positions=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 'f'),
                       indices=np.array([0, 1, 2], np.uint32),
                       material=PBRMaterial(baseColor=colour))

    def test_one_mesh_still_gives_one_node(self) -> None:
        found = self._placed(self._mesh((1.0, 0, 0))).content(self._region(), 1.0)
        assert len(found) == 1

    def test_two_meshes_give_a_node_each(self) -> None:
        rung = [self._mesh((1.0, 0, 0)), self._mesh((0, 1.0, 0))]
        assert len(self._placed(rung).content(self._region(), 1.0)) == 2

    def test_both_keep_their_own_material(self) -> None:
        rung = [self._mesh((1.0, 0, 0)), self._mesh((0, 1.0, 0))]
        found = self._placed(rung).content(self._region(), 1.0)
        assert {tuple(round(float(v), 3) for v in node.mesh.material.baseColor)
                for node in found} == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)}

    def test_they_stand_in_the_same_places(self) -> None:
        rung = [self._mesh((1.0, 0, 0)), self._mesh((0, 1.0, 0))]
        first, second = self._placed(rung).content(self._region(), 1.0)
        assert np.allclose(first.instances.translations,
                           second.instances.translations)

    def test_a_tile_with_nothing_in_it_gets_nothing(self) -> None:
        from OpenGLContext_editor.bake.bounds import BoundingBox
        rung = [self._mesh((1.0, 0, 0)), self._mesh((0, 1.0, 0))]
        empty = BoundingBox((500.0, 0.0, 500.0), (600.0, 10.0, 600.0))
        assert self._placed(rung).content(empty, 1.0) == []


class TestCombiningMeshesDoesNotLoseAMaterial:
    def _mesh(self, material):
        from OpenGLContext.scenegraph.pbrmesh import PBRMesh
        return PBRMesh(positions=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], 'f'),
                       indices=np.array([0, 1, 2], np.uint32), material=material)

    def test_meshes_of_one_material_combine(self) -> None:
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        from OpenGLContext_editor.bake.assets import combined_mesh
        material = PBRMaterial(baseColor=(1.0, 0.0, 0.0))
        combined = combined_mesh([self._mesh(material), self._mesh(material)])
        assert combined.material is material

    def test_meshes_of_different_materials_do_not(self) -> None:
        """Silently keeping the first one paints the whole prototype in it."""
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        from OpenGLContext_editor.bake.assets import combined_mesh
        with pytest.raises(ValueError):
            combined_mesh([self._mesh(PBRMaterial(baseColor=(1.0, 0.0, 0.0))),
                           self._mesh(PBRMaterial(baseColor=(0.0, 1.0, 0.0)))])

    def test_an_override_says_they_may(self) -> None:
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        from OpenGLContext_editor.bake.assets import combined_mesh
        wanted = PBRMaterial(baseColor=(0.0, 0.0, 1.0))
        combined = combined_mesh(
            [self._mesh(PBRMaterial(baseColor=(1.0, 0.0, 0.0))),
             self._mesh(PBRMaterial(baseColor=(0.0, 1.0, 0.0)))],
            material=wanted)
        assert combined.material is wanted
