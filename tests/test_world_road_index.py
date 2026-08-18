"""What a road's index costs when most of the world is nowhere near it.

Baking a world asks the road about every ground sample there is: the terrain
grid to build the earthworks, the splat map's every pixel to paint the corridor,
every candidate tree to keep the corridor clear. A four-kilometre road crosses a
sixteen-square-kilometre world; the overwhelming majority of those questions are
about ground the road never comes near, and the whole point of the index is that
they are cheap.

Grouping the queries into cells makes each *comparison* cheap. It does not make
an empty cell cheap: a two-thousand-pixel map at a five-metre cell is most of a
million groups, and iterating over them costs more than the comparisons the
index saved.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.world.road import RoadPath

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
SIDE = 2048.0


def _road(length=600.0, count=101):
    """A short road across the middle of a large world."""
    z = np.linspace(-length / 2, length / 2, count)
    return RoadPath(np.stack([np.zeros(count), np.zeros(count), z], axis=-1),
                    profile=PROFILE)


def _grid(size=512):
    axis = np.linspace(-SIDE / 2, SIDE / 2, size)
    return np.meshgrid(axis, axis)


class TestTheCostOfAskingAboutEmptyGround:
    def test_it_groups_by_where_the_road_is_not_where_the_query_is(self) -> None:
        road = _road()
        x, z = _grid()
        # The road runs down one line of a two-kilometre square. At a ten-metre
        # reach it can touch a few hundred cells; the grid has hundreds of
        # thousands.
        assert road.index_cells(x, z, radius=10.0) < 2000

    def test_and_compares_against_only_what_is_near(self) -> None:
        road = _road()
        x, z = _grid()
        assert road.comparisons(x, z, radius=10.0) < x.size

    def test_the_answer_is_the_one_it_would_have_given(self) -> None:
        """Cheap is worth nothing if it is also wrong."""
        road = _road()
        x, z = _grid(size=96)
        indexed = road.sample(x, z, radius=200.0)
        plain = road.sample(x, z)
        near = indexed.distance < 150.0
        assert np.allclose(indexed.distance[near], plain.distance[near])
        assert np.allclose(indexed.height[near], plain.height[near])

    def test_ground_the_road_never_reaches_is_infinitely_far(self) -> None:
        road = _road()
        found = road.sample(np.array([900.0]), np.array([900.0]), radius=10.0)
        assert not np.isfinite(found.distance[0])

    def test_a_query_with_no_radius_still_answers_everything(self) -> None:
        road = _road()
        found = road.sample(np.array([900.0]), np.array([900.0]))
        assert np.isfinite(found.distance[0])

    def test_a_small_query_is_not_indexed_at_all(self) -> None:
        """Below a handful of points the grouping costs more than it saves."""
        assert _road().index_cells(np.zeros(4), np.zeros(4), radius=10.0) <= 1


class TestOnTheShippedWorld:
    def test_painting_the_control_map_does_not_walk_the_world(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=1024.0, field_resolution=257,
                                control_size=512)
        circuit = world.circuit()
        axis = np.linspace(-512.0, 512.0, 512)
        x, z = np.meshgrid(axis, axis)
        assert circuit.index_cells(x, z, radius=10.0) < x.size / 50


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
