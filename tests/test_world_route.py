"""Sliding a drawn plan onto ground a road can actually follow.

A line drawn across a landscape without regard for it climbs and drops wherever
the landscape does. Held afterwards to a grade a car can drive, the alignment
then departs from the ground by whatever the difference was -- and over real
relief that is a viaduct or a bore for most of its length, which is a road
*over* a landscape rather than a road *through* one.

The fix is not to give up on the grade: it is to move the line. Sliding each
point along its own contour, towards the height its neighbours are at, finds the
route through the same country that the ground supports.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.route import REACH, ease_route


def _hill(x, z):
    """One hill in the way, a little to one side of the line drawn past it.

    To one side because a hill whose summit is exactly on the line has no
    downhill side to slide towards: the ground across the route is level there,
    and a route eased from a place like that stays where it is.
    """
    x = np.asarray(x, 'd')
    z = np.asarray(z, 'd')
    return 120.0 * np.exp(-((x / 300.0) ** 2 + ((z - 90.0) / 300.0) ** 2))


def _slope(x, z):
    """Ground falling to the east, with no line of constant height but one."""
    return 0.12 * np.asarray(x, 'd')


def _rolling(x, z):
    x = np.asarray(x, 'd')
    z = np.asarray(z, 'd')
    return 60.0 * np.sin(x / 400.0) * np.cos(z / 350.0) + 25.0 * np.sin(z / 180.0)


def _climb(plan, height_fn):
    """Total rise and fall along a plan -- what a road has to work against."""
    line = np.asarray(plan, dtype='d')
    heights = np.asarray(height_fn(line[:, 0], line[:, 1]), dtype='d')
    return float(np.abs(np.diff(heights)).sum())


def _crossing(count=80, length=1200.0):
    """A straight line running over the top of the hill."""
    return np.stack([np.linspace(-length / 2, length / 2, count),
                     np.zeros(count)], axis=-1)


def _ring(count=200, radius=700.0):
    angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


class TestItFindsGroundTheRoadCanFollow:
    def test_a_line_over_a_hill_climbs_less_afterwards(self) -> None:
        drawn = _crossing()
        eased = ease_route(drawn, _hill)
        assert _climb(eased, _hill) < _climb(drawn, _hill)

    def test_it_goes_round_the_hill_rather_than_over_it(self) -> None:
        drawn = _crossing()
        eased = ease_route(drawn, _hill)
        middle = eased[len(eased) // 3:2 * len(eased) // 3]
        assert float(middle[:, 1].mean()) < -20.0

    def test_a_summit_dead_ahead_has_no_downhill_side(self) -> None:
        """Level ground across the route is level ground across the route: the
        line stays where it was drawn rather than picking a side at random."""
        def symmetric(x, z):
            x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
            return 120.0 * np.exp(-((x / 300.0) ** 2 + (z / 300.0) ** 2))
        drawn = _crossing()
        assert np.allclose(ease_route(drawn, symmetric), drawn)

    def test_a_circuit_over_rolling_ground_climbs_less(self) -> None:
        drawn = _ring()
        eased = ease_route(drawn, _rolling, closed=True)
        assert _climb(eased, _rolling) < 0.8 * _climb(drawn, _rolling)

    def test_ground_that_is_already_level_is_left_alone(self) -> None:
        drawn = _crossing()
        eased = ease_route(drawn, lambda x, z: np.zeros(np.shape(np.asarray(x))))
        assert np.abs(eased - drawn).max() < 1.0

    def test_it_keeps_the_route_it_was_given(self) -> None:
        """A designer drew a shape, not a suggestion: the line finds better
        ground nearby rather than somewhere else entirely."""
        drawn = _ring()
        eased = ease_route(drawn, _rolling, closed=True, reach=80.0)
        assert float(np.linalg.norm(eased - drawn, axis=1).max()) <= 80.0 + 1e-6

    def test_it_may_not_move_at_all(self) -> None:
        drawn = _ring()
        assert np.allclose(ease_route(drawn, _rolling, closed=True, reach=0.0),
                           drawn)

    def test_the_reach_it_is_given_is_the_reach_it_uses(self) -> None:
        drawn = _ring()
        far = ease_route(drawn, _rolling, closed=True, reach=REACH)
        near = ease_route(drawn, _rolling, closed=True, reach=40.0)
        assert float(np.linalg.norm(far - drawn, axis=1).max()) \
            > float(np.linalg.norm(near - drawn, axis=1).max())


class TestWhatItGivesBack:
    def test_the_same_number_of_points(self) -> None:
        drawn = _ring(count=137)
        assert ease_route(drawn, _rolling, closed=True).shape == drawn.shape

    def test_it_is_a_plan_not_an_alignment(self) -> None:
        assert ease_route(_ring(), _rolling, closed=True).shape[1] == 2

    def test_an_open_route_keeps_its_ends(self) -> None:
        """A designer put them somewhere on purpose -- a junction, a start."""
        drawn = _crossing()
        eased = ease_route(drawn, _hill)
        assert np.allclose(eased[0], drawn[0])
        assert np.allclose(eased[-1], drawn[-1])

    def test_a_closed_route_stays_closed(self) -> None:
        drawn = _ring()
        eased = ease_route(drawn, _rolling, closed=True)
        step = np.linalg.norm(eased[0] - eased[-1])
        typical = np.median(np.linalg.norm(np.diff(eased, axis=0), axis=1))
        assert step < typical * 2.0

    def test_it_does_not_fold_the_line_over_itself(self) -> None:
        """A point that overtakes its neighbour makes a road that doubles back."""
        drawn = _ring()
        eased = ease_route(drawn, _rolling, closed=True)
        steps = np.linalg.norm(np.diff(np.vstack([eased, eased[:1]]), axis=0),
                               axis=1)
        assert float(steps.min()) > 0.2 * float(np.median(steps))

    def test_it_is_the_same_route_every_time(self) -> None:
        drawn = _ring()
        assert np.allclose(ease_route(drawn, _rolling, closed=True),
                           ease_route(drawn, _rolling, closed=True))

    def test_three_points_are_enough(self) -> None:
        drawn = np.array([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)])
        assert ease_route(drawn, _slope).shape == (3, 2)

    def test_two_points_have_nowhere_to_go(self) -> None:
        drawn = np.array([(0.0, 0.0), (100.0, 0.0)])
        assert np.allclose(ease_route(drawn, _slope), drawn)


class TestOnTheShippedLandscape:
    """What it is for: a circuit that is mostly road rather than mostly
    structure."""

    @pytest.fixture(scope='class')
    def landscape(self):
        """The world's own plan and its own ground, at its own relief."""
        from OpenGLContext_editor.world.procedural import (
            ProceduralWorld,
            circuit_plan,
        )
        world = ProceduralWorld(extent=4096.0)
        return (circuit_plan(world.extent * 0.32, world.extent * 0.25),
                world.natural())

    def test_the_route_climbs_a_great_deal_less(self, landscape) -> None:
        drawn, ground = landscape
        eased = ease_route(drawn, ground, reach=300.0, closed=True)
        assert _climb(eased, ground) < 0.7 * _climb(drawn, ground)

    def test_most_of_the_lap_ends_up_on_the_ground(self, landscape) -> None:
        from OpenGLContext.loaders.tiles3d.procedural import WATER_LEVEL

        from OpenGLContext_editor.world.road import follow_terrain
        from OpenGLContext_editor.world.structures import Op, choose_structures
        drawn, ground = landscape
        eased = ease_route(drawn, ground, reach=300.0, closed=True)
        line = follow_terrain(eased, ground, spacing=6.0, smoothing=60.0,
                              maximum_grade=0.10, design_speed=42.0,
                              minimum_height=WATER_LEVEL + 2.5, closed=True)
        natural = np.asarray(ground(line[:, 0], line[:, 2]), dtype='d')
        runs = choose_structures(line, natural, waterline=WATER_LEVEL,
                                 closed=True)
        total = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        laid = sum(run.length(line) for run in runs
                   if run.kind in (Op.DIRT, Op.CAUSEWAY))
        assert laid / total > 0.6


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestCornersACarCanTake:
    """Sliding a line onto easier ground puts corners into it, and a corner
    tighter than the grip available at the speed the road is for is a corner a
    car leaves. The plan is held to a radius the way the profile is held to a
    grade."""

    def _hairpin(self):
        """A line with one corner far tighter than anything can take."""
        return np.array([(-400.0, 0.0), (-200.0, 0.0), (-30.0, 0.0),
                         (0.0, 0.0), (-30.0, 40.0), (-200.0, 60.0),
                         (-400.0, 60.0)])

    def test_a_tight_corner_is_opened_out(self) -> None:
        from OpenGLContext_editor.world.route import hold_radius, least_radius
        drawn = self._hairpin()
        held = hold_radius(drawn, 150.0)
        assert least_radius(held) > least_radius(drawn)

    def test_it_is_opened_to_the_radius_it_was_given(self) -> None:
        from OpenGLContext_editor.world.route import hold_radius, least_radius
        held = hold_radius(self._hairpin(), 120.0)
        assert least_radius(held) > 100.0

    def test_a_gentle_line_is_left_alone(self) -> None:
        from OpenGLContext_editor.world.route import hold_radius
        drawn = _ring(radius=900.0)
        assert np.abs(hold_radius(drawn, 150.0, closed=True) - drawn).max() < 1.0

    def test_an_open_line_keeps_its_ends(self) -> None:
        from OpenGLContext_editor.world.route import hold_radius
        drawn = self._hairpin()
        held = hold_radius(drawn, 200.0)
        assert np.allclose(held[0], drawn[0])
        assert np.allclose(held[-1], drawn[-1])

    def test_a_straight_line_has_no_corner_at_all(self) -> None:
        from OpenGLContext_editor.world.route import least_radius
        assert not np.isfinite(least_radius(_crossing()))

    def test_the_radius_follows_the_speed_and_the_grip(self) -> None:
        from OpenGLContext_editor.world.route import cornering_radius
        assert cornering_radius(42.0, grip=1.0) == pytest.approx(179.8, abs=1.0)
        assert cornering_radius(84.0) > 3.0 * cornering_radius(42.0)

    def test_a_road_with_no_speed_has_no_limit(self) -> None:
        from OpenGLContext_editor.world.route import cornering_radius
        assert cornering_radius(0.0) == 0.0


class TestEasingKeepsTheCornersDrivable:
    def test_an_eased_route_can_be_held_to_a_radius(self) -> None:
        from OpenGLContext_editor.world.route import least_radius
        drawn = _ring()
        loose = ease_route(drawn, _rolling, closed=True)
        held = ease_route(drawn, _rolling, closed=True, minimum_radius=200.0)
        assert least_radius(held, closed=True) > least_radius(loose, closed=True)

    def test_it_still_finds_the_easier_ground(self) -> None:
        drawn = _ring()
        held = ease_route(drawn, _rolling, closed=True, minimum_radius=200.0)
        assert _climb(held, _rolling) < _climb(drawn, _rolling)

    def test_no_radius_asked_for_is_no_radius_held(self) -> None:
        drawn = _ring()
        assert np.allclose(ease_route(drawn, _rolling, closed=True),
                           ease_route(drawn, _rolling, closed=True,
                                      minimum_radius=0.0))

    def test_the_shipped_circuit_corners_at_its_design_speed(self) -> None:
        """Most of it at the design radius, and nothing so far under it that a
        car cannot take the corner at any speed worth driving."""
        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_DESIGN_SPEED,
            ProceduralWorld,
        )
        from OpenGLContext_editor.world.route import (
            _neighbours,
            _radius,
            cornering_radius,
            least_radius,
        )
        world = ProceduralWorld(extent=4096.0)
        plan = world.circuit().points[:, [0, 2]]
        wanted = cornering_radius(CIRCUIT_DESIGN_SPEED)
        before, after = _neighbours(plan, True)
        assert float(np.median(_radius(before, plan, after))) \
            >= wanted - 1.0
        assert least_radius(plan, closed=True) > 0.6 * wanted
