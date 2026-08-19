"""Water welling up at a point and running downhill from it.

Driven over height fields anyone can work out by hand -- a tilted plane, a bowl,
a pair of valleys -- so what the flow does is checked against what water does.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.height import HeightSource, edit_from_json
from OpenGLContext_editor.world.hydrology import (
    Channel,
    Spring,
    channels_for,
    flow_from,
)

EXTENT = 2000.0


def _slope(x, z):
    """Ground falling one in ten towards the east."""
    return 200.0 - np.asarray(x, dtype='d') * 0.1


def _bowl(x, z):
    """A basin with its floor at the middle, well above the waterline."""
    return 100.0 + (np.asarray(x, 'd') ** 2 + np.asarray(z, 'd') ** 2) * 1e-4


def _valley(x, z):
    """A V running north-south, so anything either side of it flows into it."""
    return 200.0 + np.abs(np.asarray(x, 'd')) * 0.2 - np.asarray(z, 'd') * 0.05


class TestWhereWaterGoes:
    def _flow(self, height_fn, at=(-800.0, 0.0), **named):
        named.setdefault('extent', EXTENT)
        return flow_from(height_fn, [Spring(at=at)], **named)[0]

    def test_it_runs_downhill_the_whole_way(self) -> None:
        path = self._flow(_slope)
        along = _slope(path.points[:, 0], path.points[:, 1])
        assert np.all(np.diff(along) <= 1e-9)

    def test_it_runs_down_the_steepest_way(self) -> None:
        """On ground tilted east, water goes east and nowhere else."""
        path = self._flow(_slope)
        assert np.all(np.diff(path.points[:, 0]) > 0.0)
        assert np.allclose(path.points[:, 1], path.points[0, 1], atol=1.0)

    def test_it_stops_at_the_edge_of_the_world(self) -> None:
        path = self._flow(_slope)
        assert path.ended == 'edge'
        assert path.points[-1, 0] == pytest.approx(EXTENT / 2.0, abs=20.0)

    def test_it_stops_when_it_reaches_the_water(self) -> None:
        """A waterline well inside the world, so reaching it is not reaching
        the edge as well."""
        path = self._flow(_slope, water_level=120.0)
        assert path.ended == 'water'
        assert _slope(*path.points[-1]) == pytest.approx(120.0, abs=5.0)

    def test_it_stops_in_a_hollow_it_cannot_get_out_of(self) -> None:
        """Which is where a lake is, and is not a failure to route."""
        path = self._flow(_bowl, at=(600.0, 0.0))
        assert path.ended == 'basin'
        assert np.hypot(*path.points[-1]) < 100.0

    def test_a_spring_on_the_flat_goes_nowhere(self) -> None:
        path = self._flow(lambda x, z: np.full(np.shape(x), 50.0))
        assert path.ended == 'basin'
        assert len(path.points) >= 1

    def test_the_path_is_walked_at_the_step_it_was_given(self) -> None:
        path = self._flow(_slope, step=25.0)
        steps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
        assert np.allclose(steps, 25.0, atol=1.0)


class TestTributaries:
    def test_two_springs_that_meet_become_one_river(self) -> None:
        springs = [Spring(at=(-400.0, 400.0)), Spring(at=(400.0, 400.0))]
        paths = flow_from(_valley, springs, extent=EXTENT)
        joined = [path for path in paths if path.ended == 'joined']
        assert len(joined) == 1

    def test_the_river_below_a_junction_carries_both(self) -> None:
        springs = [Spring(at=(-400.0, 400.0)), Spring(at=(400.0, 400.0))]
        paths = flow_from(_valley, springs, extent=EXTENT)
        main = [path for path in paths if path.ended != 'joined'][0]
        assert main.flow.max() > main.flow.min()
        assert main.flow.max() >= 2.0

    def test_springs_that_never_meet_stay_two_rivers(self) -> None:
        springs = [Spring(at=(-800.0, 800.0)), Spring(at=(-800.0, -800.0))]
        paths = flow_from(_slope, springs, extent=EXTENT)
        assert all(path.ended == 'edge' for path in paths)
        assert all(path.flow.max() == 1.0 for path in paths)

    def test_a_spring_starts_carrying_its_own_water(self) -> None:
        path = flow_from(_slope, [Spring(at=(-800.0, 0.0))], extent=EXTENT)[0]
        assert path.flow[0] == pytest.approx(1.0)


class TestTheBedItCuts:
    def _channel(self, **named):
        path = flow_from(_slope, [Spring(at=(-800.0, 0.0))], extent=EXTENT)[0]
        return channels_for([path], **named)[0]

    def test_the_ground_along_it_is_lowered(self) -> None:
        channel = self._channel()
        source = HeightSource(base=_Base(_slope), edits=[channel])
        along = channel.points[len(channel.points) // 2]
        before = float(_slope(*along))
        after = float(source.height_fn()(np.asarray([along[0]]),
                                         np.asarray([along[1]]))[0])
        assert after < before - 0.5

    def test_the_ground_away_from_it_is_not(self) -> None:
        channel = self._channel()
        source = HeightSource(base=_Base(_slope), edits=[channel])
        x = np.asarray([0.0])
        z = np.asarray([900.0])
        assert float(source.height_fn()(x, z)[0]) == pytest.approx(
            float(_slope(x, z)[0]))

    def test_it_declares_the_ground_it_can_reach(self) -> None:
        channel = self._channel()
        low_x, low_z, high_x, high_z = channel.bounds()
        assert low_x <= channel.points[:, 0].min()
        assert high_x >= channel.points[:, 0].max()

    def test_a_river_carrying_more_cuts_a_wider_bed(self) -> None:
        small = Channel(points=np.array([[0.0, -100.0], [0.0, 100.0]]),
                        flow=np.array([1.0, 1.0]))
        big = Channel(points=np.array([[0.0, -100.0], [0.0, 100.0]]),
                      flow=np.array([16.0, 16.0]))
        assert big.widths().max() > small.widths().max()

    def test_a_bed_is_deeper_where_the_river_is_bigger(self) -> None:
        channel = Channel(points=np.array([[0.0, -100.0], [0.0, 0.0],
                                           [0.0, 100.0]]),
                          flow=np.array([1.0, 4.0, 16.0]))
        assert channel.depths()[-1] > channel.depths()[0]

    def test_it_round_trips_through_a_project_file(self) -> None:
        channel = self._channel()
        again = edit_from_json(channel.to_json())
        assert np.allclose(again.points, channel.points)
        assert np.allclose(again.flow, channel.flow)

    def test_a_channel_of_one_point_carves_nothing(self) -> None:
        channel = Channel(points=np.array([[0.0, 0.0]]), flow=np.array([1.0]))
        source = HeightSource(base=_Base(_slope), edits=[channel])
        x = np.asarray([0.0])
        z = np.asarray([0.0])
        assert float(source.height_fn()(x, z)[0]) == pytest.approx(
            float(_slope(x, z)[0]))


class _Base:
    """A height base wrapping a plain function, for these tests."""

    KIND = 'test-base'

    def __init__(self, fn):
        self.fn = fn

    def sample(self, x, z):
        return np.asarray(self.fn(x, z), dtype='d')

    def to_json(self):
        return {'kind': self.KIND}

    @classmethod
    def from_json(cls, document):
        raise NotImplementedError


class TestGettingOutOfHollows:
    """Ground is not a smooth ramp: a real hillside is full of small hollows,
    and water that stopped in the first one it met would never reach anything.
    Water fills a hollow until it spills, so the routing looks past one."""

    def _dimpled(self, x, z):
        """A slope with hollows in it a step deep and a couple of steps wide."""
        x = np.asarray(x, dtype='d')
        z = np.asarray(z, dtype='d')
        return 200.0 - x * 0.1 + 6.0 * np.sin(x * 0.08) * np.cos(z * 0.08)

    def test_water_crosses_a_dimpled_slope(self) -> None:
        path = flow_from(self._dimpled, [Spring(at=(-800.0, 0.0))],
                         extent=EXTENT)[0]
        assert path.ended == 'edge'
        assert path.points[-1, 0] > 700.0

    def test_it_still_stops_in_a_hollow_it_cannot_leave(self) -> None:
        """A basin is an answer, and filling it in would be a lie about it."""
        path = flow_from(_bowl, [Spring(at=(700.0, 0.0))], extent=EXTENT)[0]
        assert path.ended == 'basin'

    def test_it_does_not_climb_out_of_the_world(self) -> None:
        path = flow_from(self._dimpled, [Spring(at=(-800.0, 0.0))],
                         extent=EXTENT)[0]
        assert np.all(np.abs(path.points) <= EXTENT / 2.0 + 1e-6)

    def test_the_way_out_of_a_hollow_is_still_downhill_overall(self) -> None:
        path = flow_from(self._dimpled, [Spring(at=(-800.0, 0.0))],
                         extent=EXTENT)[0]
        along = self._dimpled(path.points[:, 0], path.points[:, 1])
        assert along[-1] < along[0] - 100.0
