"""Obstacles in a baked world: drawn from tiles, collided from the tileset.

A prop's geometry rides in the tiles like everything else, and arrives and
leaves with them. Its *body* must not: tile geometry is level-of-detail
geometry, and a collider that came and went with a tile would be a rock a car
drives through at the moment the tile behind it swaps. So the props travel in
the tileset's ``extras`` as well, the same way the road does, and the game
stands them up itself.
"""
import json

import numpy as np
import pytest
from OpenGLContext.scenegraph.props import Prop, rock_mesh

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.props import PropLayer

REGION = BoundingBox((-500.0, -50.0, -500.0), (500.0, 50.0, 500.0))


def _props(count=6, kinds=('rock', 'boulder')):
    return [Prop(kind=kinds[index % len(kinds)],
                 position=(index * 40.0, 0.0, 0.0), yaw=0.2 * index,
                 scale=1.0 + 0.1 * index, radius=1.0, height=1.4)
            for index in range(count)]


def _layer(**named):
    named.setdefault('props', _props())
    named.setdefault('prototypes', {'rock': rock_mesh(seed=1),
                                    'boulder': rock_mesh(seed=2, radius=2.0)})
    return PropLayer(**named)


class TestWhatIsDrawn:
    def test_one_instanced_node_per_kind(self) -> None:
        assert len(_layer().content(REGION, error=1.0)) == 2

    def test_every_prop_is_placed(self) -> None:
        counted = sum(len(node.instances.translations)
                      for node in _layer().content(REGION, 1.0))
        assert counted == 6

    def test_each_is_the_size_it_was_given(self) -> None:
        for node in _layer().content(REGION, 1.0):
            assert node.instances.scales is not None

    def test_a_tile_holding_none_draws_nothing(self) -> None:
        far = BoundingBox((9000.0, -50.0, 9000.0), (9500.0, 50.0, 9500.0))
        assert _layer().content(far, 1.0) == []

    def test_a_kind_with_no_prototype_is_reported(self) -> None:
        with pytest.raises(ValueError):
            PropLayer(props=_props(), prototypes={'rock': rock_mesh()})

    def test_it_knows_the_ground_it_covers(self) -> None:
        box = _layer().bounds()
        assert box is not None and float(box.maximum[1]) > 1.0

    def test_a_world_with_no_props_has_no_bounds(self) -> None:
        assert PropLayer(props=[], prototypes={}).bounds() is None


class TestWhatIsCollided:
    def test_the_world_says_where_each_one_is(self) -> None:
        found = _layer().metadata()['props']
        assert len(found) == 6

    def test_and_how_much_room_it_takes(self) -> None:
        for one in _layer().metadata()['props']:
            assert one['radius'] > 0.0 and one['height'] > 0.0

    def test_it_reads_back_as_the_prop_it_was(self) -> None:
        first = _layer().metadata()['props'][0]
        assert Prop.from_json(first).kind == 'rock'

    def test_it_is_plain_json(self) -> None:
        assert json.loads(json.dumps(_layer().metadata()))


class TestTheShippedWorld:
    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=1024.0, field_resolution=257,
                               control_size=256)

    def test_its_verges_have_boulders_on_them(self, world) -> None:
        layer = world.prop_layer()
        assert layer is not None and len(layer.props) > 3

    def test_they_are_a_layer_of_the_bake(self, world) -> None:
        assert any(getattr(one, 'name', '') == 'props'
                   for one in world.layers())

    def test_none_of_them_is_in_the_carriageway(self, world) -> None:
        """An obstacle a driver cannot avoid is not an obstacle, it is a wall."""
        course = world.circuit()
        half = course.profile.carriageway_width / 2.0
        for prop in world.prop_layer().props:
            found = course.sample(np.array([prop.position[0]]),
                                  np.array([prop.position[2]]), radius=80.0)
            assert float(found.distance[0]) > half + prop.radius

    def test_they_stand_on_the_ground(self, world) -> None:
        ground = world.height_fn()
        for prop in world.prop_layer().props:
            at = float(np.asarray(ground(np.array([prop.position[0]]),
                                         np.array([prop.position[2]]))).ravel()[0])
            assert abs(float(prop.position[1]) - at) < 0.6

    def test_a_world_with_no_road_still_has_some(self) -> None:
        """A landscape has rocks in it whether or not anyone built a road."""
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        bare = ProceduralWorld(road=False, extent=1024.0, ground='tiles',
                               forest='tiles')
        assert bare.prop_layer() is not None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
