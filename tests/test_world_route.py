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
import math

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
        """A lap drawn across the shipped ground with no regard for it, and the
        ground it was drawn across.

        Which is what the sliding is *for*: a line put down without asking what
        is under it climbs and drops wherever the landscape does, and moving it
        along its own contours finds the route through the same country that
        the ground supports. A plan that already knows about the ground has
        nothing to be slid towards.
        """
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=4096.0)
        angle = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
        radius = 1.0 + 0.20 * np.sin(3 * angle + 3) + 0.09 * np.sin(5 * angle + 5)
        drawn = np.stack([world.extent * 0.32 * radius * np.cos(angle),
                          world.extent * 0.25 * radius * np.sin(angle)],
                         axis=-1)
        return drawn, world.natural()

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
        car cannot take the corner at any speed worth driving.

        Measured with the circuit laid out to one corner throughout
        (``variety=0``). The shipped world draws its corners from a mix instead
        -- a lap of one corner repeated is a lap a driver learns once -- and
        what that mix is bounded by is
        :data:`~OpenGLContext_editor.world.character.TIGHTEST_CORNER` rather
        than the design radius. See ``tests/test_world_variation.py``.
        """
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
        world = ProceduralWorld(extent=4096.0, variety=0.0)
        plan = world.circuit().points[:, [0, 2]]
        # The corner the *world* is laid out to, which allows for the lean its
        # bends carry: flat, the same speed would want twice the radius, and
        # that is the road banking is there to avoid having to build.
        wanted = world.corner_radius()
        assert wanted < cornering_radius(CIRCUIT_DESIGN_SPEED)
        before, after = _neighbours(plan, True)
        assert float(np.median(_radius(before, plan, after))) \
            >= wanted - 1.0
        assert least_radius(plan, closed=True) > 0.6 * wanted


class TestACircuitHasStraightsOnIt:
    """A circuit that is one continuous bend is a circuit nobody overtakes on:
    the view round a bend of radius *r* runs `sqrt(8 * r * clear)`, and a road
    cut through a wood offers a hundred-odd metres of it against the two
    hundred a pass at racing speed wants. Straights are where the passing
    happens on any real circuit, and a plan that is a modulated ellipse has
    none.
    """

    def _plan(self, **named):
        """The plan as drawn -- one point per corner."""
        from OpenGLContext_editor.world.procedural import circuit_plan
        return circuit_plan(655.0, 512.0, **named)

    def _aligned(self, **named):
        """And the same plan with its corners rounded, which is what the road
        is built along."""
        from OpenGLContext.scenegraph.road import cornering_radius

        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_DESIGN_SPEED,
            CIRCUIT_SPACING,
        )
        from OpenGLContext_editor.world.route import hold_corners
        return hold_corners(self._plan(**named),
                            minimum=cornering_radius(CIRCUIT_DESIGN_SPEED),
                            closed=True, spacing=CIRCUIT_SPACING)

    @staticmethod
    def _radii(plan):
        before, after = np.roll(plan, 1, axis=0), np.roll(plan, -1, axis=0)
        first = np.linalg.norm(plan - before, axis=1)
        second = np.linalg.norm(after - plan, axis=1)
        third = np.linalg.norm(after - before, axis=1)
        side, other = plan - before, after - before
        area2 = np.abs(side[:, 0] * other[:, 1] - side[:, 1] * other[:, 0])
        return np.where(area2 < 1e-9, 1e6,
                        first * second * third / (2.0 * np.maximum(area2, 1e-30)))

    @staticmethod
    def _share(plan, keep):
        """How much of the plan's *length* the kept points carry.

        By length rather than by count: an arc is written down every few metres
        and a straight is two points however long it is, so counting points
        says a circuit of straights is all corner.
        """
        step = np.linalg.norm(np.roll(plan, -1, axis=0) - plan, axis=1)
        run = 0.5 * (step + np.roll(step, 1))
        return float(run[keep].sum() / run.sum())

    def test_a_good_part_of_it_is_straight(self) -> None:
        """Straight enough that a driver sees to the end of what they are
        planning over, which is a bend of some hundreds of metres."""
        aligned = self._aligned()
        straight = self._share(aligned, self._radii(aligned) > 2000.0)
        assert straight > 0.25, 'only %.0f%% of the lap is straight' % (
            100.0 * straight,)

    def test_and_the_straights_are_joined_by_corners(self) -> None:
        """The plan is a polygon -- a corner is one vertex of it, and what
        turns that into something a car can take is the alignment rounding it
        to the radius the design speed asks for. So the corners are counted on
        the *aligned* plan, which is what the road gets built along.
        """
        from OpenGLContext.scenegraph.road import cornering_radius

        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_CORNERS,
            CIRCUIT_DESIGN_SPEED,
        )
        wanted = cornering_radius(CIRCUIT_DESIGN_SPEED)
        radii = self._radii(self._aligned())
        assert (radii < 4.0 * wanted).sum() >= CIRCUIT_CORNERS
        assert radii.min() > 0.5 * wanted, (
            'a corner of %.0f m where %.0f m was asked for'
            % (radii.min(), wanted))

    def test_it_still_closes_on_itself(self) -> None:
        """The last point runs on to the first, whatever the legs between the
        corners are written down as."""
        plan = self._aligned()
        closing = float(np.linalg.norm(plan[0] - plan[-1]))
        legs = np.linalg.norm(np.roll(plan, -1, axis=0) - plan, axis=1)
        assert closing <= legs.max()

    def test_and_goes_round_once(self) -> None:
        plan = self._aligned()
        step = np.diff(np.vstack([plan, plan[:2]]), axis=0)
        angle = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
        assert abs(angle[-1] - angle[0]) == pytest.approx(2.0 * math.pi, abs=0.3)

    def test_a_caller_can_ask_for_more_corners_or_fewer(self) -> None:
        assert len(self._plan(corners=4)) == 4
        assert len(self._plan(corners=9)) == 9


class TestTheWorldsOwnCircuitIsBuiltAsDrawn:
    """Sliding a line sideways onto ground it can follow is for a route that
    was *found*, where the shape is an artefact of the search. The world's own
    circuit is drawn: straights joined by corners of the radius its design
    speed asks for, and the straights are what it is overtaken on.

    Slid, they come back as gentle curves and the overtaking goes with them --
    and on this landscape the sliding buys nothing to pay for it: the same
    structures over the same share of the lap, the same earthwork, the same
    grade.
    """

    def _circuit(self, **named):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=2048.0, seed=11, **named)
        return world, np.asarray(world.circuit().points, dtype='d')

    @staticmethod
    def _straight(line):
        xz = line[:, [0, 2]]
        before, after = np.roll(xz, 1, axis=0), np.roll(xz, -1, axis=0)
        first = np.linalg.norm(xz - before, axis=1)
        second = np.linalg.norm(after - xz, axis=1)
        third = np.linalg.norm(after - before, axis=1)
        side, other = xz - before, after - before
        area2 = np.abs(side[:, 0] * other[:, 1] - side[:, 1] * other[:, 0])
        radii = np.where(area2 < 1e-9, 1e6,
                         first * second * third / (2.0 * np.maximum(area2, 1e-30)))
        return float((radii > 2000.0).mean())

    def test_a_good_part_of_the_lap_is_still_straight(self) -> None:
        _world, line = self._circuit()
        assert self._straight(line) > 0.25

    def test_which_sliding_it_would_have_taken_away(self) -> None:
        _world, slid = self._circuit(ease=True)
        assert self._straight(slid) < 0.1

    def test_and_the_sliding_was_not_paying_for_itself(self) -> None:
        """Same structures, same earthwork: the road sits on this landscape as
        well drawn as slid."""
        drawn, drawn_line = self._circuit()
        slid, slid_line = self._circuit(ease=True)
        for world, line in ((drawn, drawn_line), (slid, slid_line)):
            ground = np.asarray(world.natural()(line[:, 0], line[:, 2]),
                                dtype='d').ravel()
            world.worst = float(np.percentile(np.abs(line[:, 1] - ground), 90))
        assert drawn.worst < 1.3 * slid.worst


class TestTheTightestCornerIsMeasuredOverRoad:
    """The circle through a point and its two neighbours is exact for a circle
    and noisy for a road. An alignment written down every few metres carries
    its arcs as chords, so a point a hand's breadth off its arc reads as a
    corner half the radius of the one it is on -- and a check on the tightest
    corner in a plan then fails a road that is fine.
    """

    def _arc(self, radius=200.0, spacing=6.0, jitter=0.0, seed=7):
        angle = np.arange(0.0, 2.0 * math.pi, spacing / radius)
        line = np.stack([radius * np.cos(angle), radius * np.sin(angle)],
                        axis=-1)
        if jitter:
            wobble = np.random.default_rng(seed).normal(0.0, jitter, len(line))
            line *= (1.0 + wobble / radius)[:, None]
        return line

    def test_a_clean_bend_reads_as_the_bend_it_is(self) -> None:
        from OpenGLContext_editor.world.route import least_radius
        assert least_radius(self._arc(200.0), closed=True) == pytest.approx(
            200.0, rel=0.05)

    def test_and_one_written_down_untidily_still_does(self) -> None:
        from OpenGLContext_editor.world.route import least_radius
        found = least_radius(self._arc(200.0, jitter=0.1), closed=True)
        assert found > 0.6 * 200.0, '%.0f m for a 200 m bend' % found

    def test_a_tighter_bend_still_reads_tighter(self) -> None:
        from OpenGLContext_editor.world.route import least_radius
        assert least_radius(self._arc(80.0), closed=True) < \
            least_radius(self._arc(400.0), closed=True)

    def test_and_a_hairpin_is_still_a_hairpin(self) -> None:
        """Short enough to resolve the corner it is about: a plan with one
        tight corner in it does not have that corner averaged away."""
        from OpenGLContext_editor.world.route import least_radius
        drawn = np.array([(-400.0, 0.0), (-200.0, 0.0), (-30.0, 0.0),
                          (0.0, 0.0), (-30.0, 40.0), (-200.0, 60.0),
                          (-400.0, 60.0)])
        assert least_radius(drawn) < 60.0
