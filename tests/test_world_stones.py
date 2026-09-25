"""The loose stone a world strews over its hillsides.

Part of the ground rather than an obstacle in it: drawn in the tiles close
enough to show it, and solid everywhere, so walking onto one puts you on top of
it. The table a game stands up travels in the tileset beside the boulders'.
"""
import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.stones import StoneLayer
from OpenGLContext_editor.world.procedural import ProceduralWorld

EXTENT = 512.0


def _world(**kwargs):
    kwargs.setdefault('tree_density', 0.0)
    return ProceduralWorld(extent=EXTENT, field_resolution=129,
                           control_size=128, **kwargs)


class TestWhereTheStoneIs:
    def test_a_world_has_some(self) -> None:
        assert len(_world(road=False).stones().positions) > 0

    def test_it_lies_inside_the_world(self) -> None:
        points = _world(road=False).stones().positions
        assert float(np.abs(points[:, [0, 2]]).max()) <= EXTENT / 2.0 + 1e-6

    def test_and_above_the_waterline(self) -> None:
        world = _world(road=False)
        assert float(world.stones().positions[:, 1].min()) > world.water_level

    def test_none_of_it_is_on_the_road(self) -> None:
        """Not on the verge or the shoulder either. Those were graded when the
        road was built, and they are what a car that has run wide recovers on:
        an obstacle there is one a driver has nothing left to avoid it with."""
        world = _world(road=True)
        points = world.stones().positions
        circuit = world.circuit()
        found = circuit.sample(points[:, 0], points[:, 2], radius=60.0)
        near = np.asarray(found.distance)
        assert float(np.nanmin(np.where(np.isfinite(near), near, np.inf))) \
            > circuit.widest() / 2.0

    def test_and_none_of_it_is_bigger_than_a_wheel_rides_over(self) -> None:
        """Loose stone is what a hillside is made of. A rock the height of a
        wheel is a boulder, and a boulder is placed deliberately, off the road,
        as something to be stopped by."""
        world = _world(road=False)
        assert float(np.max(world.stones().scales)) <= 0.4

    def test_a_world_with_no_stone_has_no_layer(self) -> None:
        world = _world(road=False)
        world.stones = lambda: _Empty()
        assert world.stone_layer() is None


class _Empty:
    positions = np.zeros((0, 3))
    yaws = np.zeros(0)
    scales = np.zeros(0)


class TestTheLayerItBecomes:
    def test_the_world_lists_it_among_its_layers(self) -> None:
        names = [getattr(layer, 'name', None)
                 for layer in _world(road=False).layers()]
        assert 'stones' in names

    def test_it_is_seated_on_the_ground_the_world_is_collided_against(self) -> None:
        """One surface for both: the grain the tiles draw is in the landscape,
        and a stone is seated on that, so what is drawn and what is stood on
        are the same number."""
        world = _world(road=False, ground='tiles')
        layer = world.stone_layer()
        assert isinstance(layer, StoneLayer)
        ground = world.landscape().field()
        feet = np.asarray([one.position for one in layer.stones])[:200]
        under = np.asarray(ground.sample(feet[:, 0], feet[:, 2]))
        # Bedded into the hillside by a share of their own size, and no more.
        sunk = under - feet[:, 1]
        assert float(sunk.min()) >= -0.05
        assert float(sunk.max()) <= 0.3

    def test_a_close_tile_carries_stone_and_a_distant_one_does_not(self) -> None:
        layer = _world(road=False, ground='tiles').stone_layer()
        tile = BoundingBox((-32.0, -1000.0, -32.0), (32.0, 1000.0, 32.0))
        assert layer.content(tile, error=60.0) == []
        assert layer.content(tile, error=1.0) != []

    def test_the_stone_is_cut_from_a_handful_of_shapes(self) -> None:
        from OpenGLContext_editor.world.procedural import STONE_SHAPES
        layer = _world(road=False).stone_layer()
        assert len(layer.prototypes) == STONE_SHAPES
        assert len(layer.kinds()) == STONE_SHAPES


class TestStandingOnOne:
    """The point of the whole thing. A stone lying in the grass is ground: walk
    onto one and you are on top of it."""

    def _layer(self):
        return _world(road=False, ground='tiles').stone_layer()

    def test_every_stone_travels_for_a_game_to_stand_up(self) -> None:
        from OpenGLContext.scenegraph.props import props_from_table
        layer = self._layer()
        record = layer.metadata()['stones']
        assert record['count'] == len(layer.stones)
        assert len(props_from_table(layer.assets()[record['table']])) \
            == len(layer.stones)

    def test_each_is_a_dome_rather_than_a_block(self) -> None:
        """A block the size of a stone is a kerb across the hillside."""
        assert {one.shape for one in self._layer().stones} == {'dome'}

    def test_it_is_measured_from_the_shape_it_is_drawn_as(self) -> None:
        layer = self._layer()
        one = layer.stones[0]
        drawn = layer.prototypes[one.kind]
        tall = float(np.ptp(np.asarray(drawn.positions)[:, 1])) * one.scale
        assert one.height == pytest.approx(tall, rel=1e-4)
        assert 0.0 < one.radius <= one.scale * 1.2

    def test_and_a_body_stands_where_the_stone_is_drawn(self) -> None:
        """Built through the engine's own collider, so what the game does with
        the table is what is asserted rather than what the table says."""
        from omi_physics.world import PhysicsWorld
        from OpenGLContext.physics.props import PropColliders
        from OpenGLContext.scenegraph.props import props_from_table
        layer = self._layer()
        stones = props_from_table(layer.assets()['stones.npz'])
        one = min(stones, key=lambda s: float(np.hypot(s.position[0],
                                                       s.position[2])))
        world = PhysicsWorld()
        PropColliders(world, [one], reach=10.0).update(one.position)
        assert world.live_body_count == 1
        top = float(world.position[0][1]) + float(
            world.shapes[world.collider_shape[0]].radius)
        assert top == pytest.approx(one.position[1] + one.height, abs=0.01)

    def test_and_a_re_bake_strews_the_same_stone(self) -> None:
        one = _world(road=False, seed=5).stones().positions
        two = _world(road=False, seed=5).stones().positions
        assert np.array_equal(one, two)

    def test_a_different_world_gets_different_stone(self) -> None:
        one = _world(road=False, seed=5).stones().positions
        two = _world(road=False, seed=6).stones().positions
        assert not np.array_equal(one[:10], two[:10])
