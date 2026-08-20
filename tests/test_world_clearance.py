"""How far the forest is held back from a road that is not on the ground.

On the ground a crown reaching over the carriageway is the point: that is the
canopy closing over a forest road. Where the road is *carried* -- an embankment,
a causeway's retained fill, a deck -- a tree standing beside it is rooted metres
below the surface, and a crown that reaches the same distance goes through the
structure instead of over the road. So the clearance is the corridor where the
road is on the land and the corridor plus a crown where it is above it.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.world.procedural import (
    CROWN_RADIUS,
    ROAD_CLEARANCE,
    ProceduralWorld,
)
from OpenGLContext_editor.world.road import RoadPath
from OpenGLContext_editor.world.structures import Op

PROFILE = RoadProfile(lane_width=3.6, lanes=2, shoulder_width=0.7,
                      shoulder_drop=0.05, verge_width=1.0, verge_drop=0.35)


class _World(ProceduralWorld):
    """A world whose road and ground are stated rather than generated.

    ``lift`` carries the road that far over the land on a causeway; ``sink``
    puts it that far under it in a bore.
    """

    def __init__(self, lift=0.0, sink=0.0, **named):
        super().__init__(**named)
        self._lift = lift
        self._sink = sink

    def natural(self):
        return lambda x, z: np.zeros(np.shape(np.asarray(x, 'd')))

    def circuit(self):
        if self._circuit is None:
            z = np.linspace(-500.0, 500.0, 201)
            height = self._lift - self._sink
            line = np.stack([np.zeros(201), np.full(201, height), z], axis=-1)
            op = (Op.TUNNEL if self._sink
                  else Op.CAUSEWAY if self._lift else Op.DIRT)
            ops = np.full(201, op, dtype=object)
            self._circuit = RoadPath(line, profile=PROFILE, ops=ops)
        return self._circuit


def _kept(world, offsets):
    points = np.stack([np.asarray(offsets, 'd'),
                       np.zeros(len(offsets)),
                       np.zeros(len(offsets))], axis=-1)
    return np.asarray(world._away_from_the_road(points))


class TestInsideABore:
    """Nothing grows in a tunnel.

    The ground a bore runs through is taken out from under it, down to the
    road, for as far as the bore goes -- so the land a tree would have stood on
    is not there any more, and one planted on the cut floor is a tree inside
    the tunnel, in plain view through the portal. The cleared corridor is the
    whole of what was dug out.
    """

    def _world(self, sink=30.0):
        return _World(extent=1024.0, sink=sink)

    def test_a_tree_where_the_hill_was_is_dropped(self) -> None:
        world = self._world()
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert not bool(_kept(world, [corridor + 5.0])[0])

    def test_one_out_past_the_cutting_is_kept(self) -> None:
        world = self._world(sink=30.0)
        assert bool(_kept(world, [200.0])[0])

    def test_it_is_the_same_on_both_sides(self) -> None:
        world = self._world()
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert list(_kept(world, [-(corridor + 5.0), corridor + 5.0])) \
            == [False, False]

    def test_a_shallower_bore_clears_less(self) -> None:
        shallow = float(np.sum(_kept(self._world(sink=10.0),
                                     np.linspace(10.0, 200.0, 96))))
        deep = float(np.sum(_kept(self._world(sink=40.0),
                                  np.linspace(10.0, 200.0, 96))))
        assert shallow > deep


class TestOnTheGround:
    def test_a_tree_at_the_verge_is_kept(self) -> None:
        world = _World(extent=1024.0)
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert bool(_kept(world, [corridor + 0.2])[0])

    def test_one_in_the_carriageway_is_not(self) -> None:
        assert not bool(_kept(_World(extent=1024.0), [1.0])[0])

    def test_the_canopy_may_still_reach_over_the_road(self) -> None:
        """Which is the point of a forest road."""
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert corridor < PROFILE.total_width / 2.0 + CROWN_RADIUS


class TestWhereTheRoadIsCarried:
    def test_a_tree_at_the_verge_is_dropped(self) -> None:
        """It would grow through the side of the fill, not over the road."""
        world = _World(lift=8.0, extent=1024.0)
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert not bool(_kept(world, [corridor + 0.2])[0])

    def test_one_a_crown_further_out_is_kept(self) -> None:
        world = _World(lift=8.0, extent=1024.0)
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert bool(_kept(world, [corridor + CROWN_RADIUS + 0.5])[0])

    def test_a_road_a_hand_above_the_ground_is_still_on_it(self) -> None:
        """Every road stands a little proud of the land it is built on."""
        world = _World(lift=0.4, extent=1024.0)
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert bool(_kept(world, [corridor + 0.2])[0])

    def test_it_is_the_same_on_both_sides(self) -> None:
        world = _World(lift=8.0, extent=1024.0)
        corridor = PROFILE.total_width / 2.0 + ROAD_CLEARANCE
        assert list(_kept(world, [corridor + 0.2, -(corridor + 0.2)])) \
            == [False, False]


class TestTheShippedWorld:
    @pytest.fixture(scope='class')
    def world(self):
        return ProceduralWorld(extent=1024.0, field_resolution=257,
                               control_size=256)

    def test_no_tree_grows_through_a_structure(self, world) -> None:
        """A crown's reach from the trunk, against how far the built road
        stands out where that trunk is."""
        circuit = world.circuit()
        trees = world.scatter().positions
        ground = np.asarray(world.natural()(trees[:, 0], trees[:, 2]), 'd')
        found = circuit.sample(trees[:, 0], trees[:, 2], radius=90.0)
        carried = found.height - ground > 2.0
        near = found.distance < circuit.profile.total_width / 2.0 + CROWN_RADIUS
        assert not bool(np.any(carried & near))

    def test_the_forest_still_comes_up_to_the_road(self, world) -> None:
        circuit = world.circuit()
        trees = world.scatter().positions
        found = circuit.sample(trees[:, 0], trees[:, 2], radius=90.0)
        assert float(found.distance.min()) \
            < circuit.profile.total_width / 2.0 + ROAD_CLEARANCE + 1.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
