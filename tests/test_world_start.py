"""Where a lap begins, and how a game finds out.

The gantry, the grid and the lap timing all read one number: how far along the
circuit the line is drawn. A designer chooses where that is, so it travels from
the route, through the generator, into the tileset a game opens.
"""
import math

import numpy as np
import pytest

from OpenGLContext_editor.world.procedural import ProceduralWorld


def _circuit(points=24, radius=380.0):
    return np.asarray([(radius * math.cos(2 * math.pi * i / points),
                        radius * 0.75 * math.sin(2 * math.pi * i / points))
                       for i in range(points)], dtype='d')


def _world(**named):
    named.setdefault('extent', 1024.0)
    named.setdefault('resolution', 17)
    named.setdefault('tree_density', 0.0)
    named.setdefault('route', _circuit())
    return ProceduralWorld(**named)


class TestWhereTheLapBegins:
    def test_with_nothing_asked_for_it_is_where_the_line_starts(self) -> None:
        assert _world().start_station() == pytest.approx(0.0)

    def test_a_chosen_point_puts_it_there(self) -> None:
        plan = _circuit()
        world = _world(route=plan, start_at=tuple(plan[6]))
        assert world.start_station() > 0.0

    def test_it_is_the_station_nearest_the_point_asked_for(self) -> None:
        """The generated centreline is not the drawn plan -- it is smoothed and
        re-sampled -- so the choice is carried by where it is, not by an index
        into a list that no longer exists."""
        plan = _circuit()
        world = _world(route=plan, start_at=tuple(plan[6]))
        path = world.circuit()
        at = path.points[int(np.searchsorted(path.stations,
                                             world.start_station()))]
        assert np.hypot(at[0] - plan[6][0], at[2] - plan[6][1]) < 60.0

    def test_the_gantry_stands_at_the_line(self) -> None:
        plan = _circuit()
        world = _world(route=plan, start_at=tuple(plan[12]))
        gantry = world.start_line()
        assert np.hypot(gantry.position[0] - plan[12][0],
                        gantry.position[2] - plan[12][1]) < 60.0

    def test_a_point_nowhere_near_the_line_still_lands_on_it(self) -> None:
        world = _world(start_at=(0.0, 0.0))
        assert 0.0 <= world.start_station() <= world.circuit().length


class TestWhatTheGameIsTold:
    def test_the_road_says_where_the_lap_begins(self) -> None:
        plan = _circuit()
        world = _world(route=plan, start_at=tuple(plan[6]))
        road = world.circuit_layer().metadata()['roads'][0]
        assert road['start'] == pytest.approx(world.start_station())

    def test_it_is_a_distance_along_the_centreline_it_gives(self) -> None:
        world = _world()
        road = world.circuit_layer().metadata()['roads'][0]
        assert 0.0 <= road['start'] <= road['length']
