"""Loose stone: drawn in the tiles close enough to show it, solid everywhere.

What a hillside is made of, as against what a car runs into. A boulder is an
obstacle and is held for hundreds of metres around; a stone is ground -- a few
kilobytes of placements in the tile that draws it, and a dome in the physics
world where a wheel is.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.props import Prop, rock_mesh

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.stones import StoneLayer

SHAPES = {'stone%d' % n: rock_mesh(radius=1.0, seed=n) for n in range(2)}


def _placed(nodes):
    """Every stone placement across a tile's nodes, as (N,3) positions."""
    if not nodes:
        return np.zeros((0, 3))
    return np.vstack([node.instances.translations for node in nodes])


def _stones(count=12, radius=0.4, spread=40.0):
    at = np.linspace(-spread, spread, count)
    kinds = sorted(SHAPES)
    return StoneLayer(
        stones=[Prop(kind=kinds[index % len(kinds)],
                     position=(float(where), float(where) / 20.0, 0.0),
                     yaw=float(index) * 0.5, scale=radius,
                     radius=radius, height=radius * 1.1, shape='dome')
                for index, where in enumerate(at)],
        prototypes=SHAPES)


def _region(low=-50.0, high=50.0):
    return BoundingBox((low, -500.0, low), (high, 500.0, high))


class TestWhereLooseStoneIs:
    def test_its_bounds_hold_every_stone(self) -> None:
        box = _stones().bounds()
        assert box.minimum[0] <= -40.0 and box.maximum[0] >= 40.0

    def test_a_layer_with_no_stone_has_no_bounds(self) -> None:
        empty = StoneLayer(stones=[], prototypes=SHAPES)
        assert empty.bounds() is None
        assert empty.content(_region(), error=0.1) == []

    def test_only_the_stones_in_the_tile_are_written(self) -> None:
        found = _placed(_stones().content(_region(-50.0, 0.0), error=0.1))
        assert float(found[:, 0].max()) <= 0.001

    def test_a_tile_holding_none_of_them_writes_nothing(self) -> None:
        assert _stones().content(_region(200.0, 300.0), error=0.1) == []

    def test_a_stone_with_no_shape_to_wear_is_reported(self) -> None:
        with pytest.raises(ValueError):
            StoneLayer(stones=[Prop(kind='pebble', position=(0, 0, 0))],
                       prototypes=SHAPES)


class TestWhichLevelsDrawIt:
    def test_a_tile_too_coarse_to_show_a_stone_leaves_it_out(self) -> None:
        """Half a metre of rock drawn on a tile that may be twenty metres wrong
        is bandwidth spent on something nobody can see."""
        assert _stones(radius=0.4).content(_region(), error=20.0) == []
        assert _stones(radius=0.4).detail == pytest.approx(5.0)

    def test_a_tile_fine_enough_carries_it(self) -> None:
        assert _stones(radius=0.4).content(_region(), error=0.2) != []

    def test_and_the_bigger_stone_arrives_first(self) -> None:
        """Refining the tree brings the ground in by size, so a hillside fills
        in rather than switching on."""
        error = 0.6
        assert _stones(radius=1.0).content(_region(), error) != []
        assert _stones(radius=0.1).content(_region(), error) == []

    def test_how_soon_is_the_layer_s_to_say(self) -> None:
        eager = _stones(radius=0.4)
        eager.detail = 100.0
        assert eager.content(_region(), error=20.0) != []


class TestWhatOneTileMayDraw:
    """The tree refines with REPLACE, so a tile stands in for its whole subtree
    and a stone fine enough to draw is written into every level from there
    down. A coarse tile is thinned to a stand-in, the way an instance layer's
    scatter is, so the duplication is bounded."""

    def test_a_tile_is_capped_at_what_it_was_given(self) -> None:
        stones = _stones(count=40)
        stones.max_stones = 5
        assert len(_placed(stones.content(_region(), error=0.1))) == 5

    def test_the_thinning_stays_spread_over_the_tile(self) -> None:
        stones = _stones(count=40, spread=40.0)
        stones.max_stones = 4
        found = _placed(stones.content(_region(), error=0.1))
        assert float(found[:, 0].min()) < -20.0
        assert float(found[:, 0].max()) > 20.0

    def test_and_is_the_same_every_bake(self) -> None:
        one, two = _stones(count=40), _stones(count=40)
        one.max_stones = two.max_stones = 7
        assert np.array_equal(_placed(one.content(_region(), 0.1)),
                              _placed(two.content(_region(), 0.1)))

    def test_a_tile_inside_the_cap_carries_the_lot(self) -> None:
        stones = _stones(count=6)
        stones.max_stones = 100
        assert len(_placed(stones.content(_region(), error=0.1))) == 6

    def test_but_the_thinning_never_reaches_what_is_stood_up(self) -> None:
        """A stone a thinned tile left out is still something to stand on."""
        stones = _stones(count=40)
        stones.max_stones = 4
        assert len(stones.metadata()['stones']) == 40


class TestWhatATileGets:
    """One node per shape, however much stone the tile holds: a tile of two
    hundred stones is the shapes it already carries plus two hundred
    placements, not two hundred stones' worth of triangles."""

    def test_a_node_for_each_shape_and_no_more(self) -> None:
        found = _stones(count=12).content(_region(), error=0.1)
        assert len(found) == len(SHAPES)

    def test_the_shapes_are_the_prototypes_themselves(self) -> None:
        found = _stones().content(_region(), error=0.1)
        assert {id(node.mesh) for node in found} \
            == {id(mesh) for mesh in SHAPES.values()}

    def test_every_stone_is_placed_once(self) -> None:
        assert len(_placed(_stones(count=16).content(_region(), 0.1))) == 16

    def test_each_is_scaled_to_its_own_size(self) -> None:
        found = _stones(count=4, radius=0.7).content(_region(), 0.1)
        assert np.allclose(found[0].instances.scales, 0.7)

    def test_and_turned_to_its_own_heading(self) -> None:
        found = _stones(count=8).content(_region(), 0.1)
        turns = np.vstack([n.instances.rotations for n in found])
        assert len(np.unique(turns[:, 1])) > 1
        assert np.allclose(np.linalg.norm(turns, axis=1), 1.0, atol=1e-5)

    def test_it_is_drawn_where_it_is_stood_up(self) -> None:
        """One number for both: the placement's translation is the prop's own
        position, so nothing has to be kept in step afterwards."""
        stones = _stones(count=4)
        drawn = _placed(stones.content(_region(), 0.1))
        stood = np.asarray([one['at'] for one in stones.metadata()['stones']])
        assert np.allclose(np.sort(drawn[:, 0]), np.sort(stood[:, 0]),
                           atol=1e-3)


class TestTheTableAGameStandsUp:
    def test_every_stone_is_in_it(self) -> None:
        assert len(_stones(count=9).metadata()['stones']) == 9

    def test_each_is_a_dome_rather_than_a_block(self) -> None:
        """A stone is ground: a wheel rides over one. A block the size of it is
        a kerb across the hillside."""
        for one in _stones().metadata()['stones']:
            assert one['shape'] == 'dome'

    def test_and_reads_back_as_the_prop_it_was(self) -> None:
        stones = _stones(count=3)
        read = [Prop.from_json(one) for one in stones.metadata()['stones']]
        assert [one.position for one in read] \
            == [tuple(one.position) for one in stones.stones]

    def test_it_is_its_own_channel_and_not_the_world_s_props(self) -> None:
        """The two are held at different reaches, so a game that merged them
        would carry every stone in the world as far as it carries a boulder."""
        assert 'props' not in _stones().metadata()
