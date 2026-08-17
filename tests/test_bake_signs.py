"""Signs in a baked world: one instanced draw per kind, one picture per kind.

A world has tens of signs and a handful of kinds, so each kind is one prototype
placed many times rather than a mesh each. The plate's picture is written once
beside the tileset and named by every tile that carries that kind, the way the
road surface is: embedded per tile it would arrive again with every tile.
"""
import json

import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.signs import SIGN_DIRECTORY, SignLayer
from OpenGLContext_editor.world.signs import Placement

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
REGION = BoundingBox((-500.0, -50.0, -500.0), (500.0, 50.0, 500.0))


def _placed(count=4, kinds=('bend-left', 'dip')):
    return [Placement(position=np.array([i * 30.0, 0.0, 0.0]), yaw=0.3 * i,
                      kind=kinds[i % len(kinds)]) for i in range(count)]


def _layer(**named):
    named.setdefault('placements', _placed())
    return SignLayer(**named)


class TestWhatItWrites:
    def test_one_node_per_kind_per_part(self) -> None:
        """A post and a plate for each kind: two materials, two nodes."""
        found = _layer().content(REGION, error=1.0)
        assert len(found) == 4

    def test_a_kind_with_no_signs_is_not_written(self) -> None:
        found = _layer(placements=_placed(kinds=('dip',))).content(REGION, 1.0)
        assert len(found) == 2

    def test_every_sign_is_placed(self) -> None:
        counted = sum(len(node.instances.translations)
                      for node in _layer().content(REGION, 1.0))
        assert counted == 8            # four signs, a post and a plate each

    def test_a_tile_holding_none_writes_nothing(self) -> None:
        far = BoundingBox((5000.0, -50.0, 5000.0), (6000.0, 50.0, 6000.0))
        assert _layer().content(far, 1.0) == []

    def test_a_distant_tile_leaves_them_out(self) -> None:
        """A sign is a metre across; a tile whose error is tens of metres
        cannot show it and should not carry it."""
        assert _layer().content(REGION, error=40.0) == []

    def test_it_knows_the_ground_it_covers(self) -> None:
        box = _layer().bounds()
        assert box is not None
        assert float(box.maximum[1]) > 2.0          # the plate stands up

    def test_a_world_with_no_signs_has_no_bounds(self) -> None:
        assert SignLayer(placements=[]).bounds() is None


class TestThePictures:
    def test_each_kind_writes_its_plate_once(self) -> None:
        assets = _layer().assets()
        assert sorted(assets) == ['%s/bend-left.png' % SIGN_DIRECTORY,
                                  '%s/dip.png' % SIGN_DIRECTORY]

    def test_they_are_real_images(self) -> None:
        import io

        from PIL import Image
        for data in _layer().assets().values():
            assert Image.open(io.BytesIO(data)).size[0] > 32

    def test_a_tile_names_the_file_rather_than_carrying_it(self) -> None:
        for node in _layer().content(REGION, 1.0):
            texture = node.mesh.material.texture('baseColor')
            if texture is not None:
                assert texture.uri.startswith(SIGN_DIRECTORY + '/')

    def test_the_world_says_where_its_signs_are(self) -> None:
        found = _layer().metadata()['signs']
        assert len(found) == 4
        assert {one['kind'] for one in found} == {'bend-left', 'dip'}

    def test_that_metadata_is_plain_json(self) -> None:
        assert json.loads(json.dumps(_layer().metadata()))


class TestTheShippedWorld:
    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        # A quarter of the shipped world's side, which is the same landscape and
        # the same generator at a resolution a test can afford.
        return ProceduralWorld(extent=1024.0, field_resolution=257,
                               control_size=256)

    def test_its_circuit_is_signed(self, world) -> None:
        layer = world.sign_layer()
        assert layer is not None and len(layer.placements) > 3

    def test_the_signs_are_a_layer_of_the_bake(self, world) -> None:
        assert any(getattr(one, 'name', '') == 'signs' for one in world.layers())

    def test_a_world_with_no_road_has_none(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        assert ProceduralWorld(road=False).sign_layer() is None

    def test_they_stand_beside_the_road_not_on_it(self, world) -> None:
        course = world.circuit()
        half = course.profile.total_width / 2.0
        for placed in world.sign_layer().placements:
            found = course.sample(np.array([placed.position[0]]),
                                  np.array([placed.position[2]]),
                                  radius=60.0)
            assert float(found.distance[0]) > half


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestASignStandsWhereItCanBeSeen:
    """A forest road is cleared only as wide as it has to be, and a sign put
    outside that strip is a sign standing in the trees. The world knows both
    numbers -- how far the trees are held back and how far a post stands out --
    so it is the world's job to keep the second inside the first."""

    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=1024.0, field_resolution=257,
                               control_size=256)

    def test_every_sign_is_inside_the_cleared_corridor(self, world) -> None:
        corridor = world._corridor()
        course = world.circuit()
        for placed in world.sign_layer().placements:
            found = course.sample(np.array([placed.position[0]]),
                                  np.array([placed.position[2]]), radius=60.0)
            assert float(found.distance[0]) < corridor

    def test_no_tree_stands_where_a_sign_does(self, world) -> None:
        trees = world.scatter().positions
        for placed in world.sign_layer().placements:
            gap = np.hypot(trees[:, 0] - placed.position[0],
                           trees[:, 2] - placed.position[2])
            assert float(gap.min()) > 0.5
