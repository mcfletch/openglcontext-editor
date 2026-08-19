"""A road that climbs by turning back on itself.

A switchback is a hairpin drawn on purpose: the only way to gain height on a
slope too steep to climb straight up. What it needs from the generator is to be
*kept* -- rounded to a corner a car can take, but not opened out into a sweep
that goes somewhere else.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.route import (
    hold_corners,
    hold_radius,
    least_radius,
)


def _hairpin(leg=300.0, offset=120.0):
    """Up the slope, back on itself, and up again."""
    return np.array([(0.0, 0.0), (leg, 0.0), (leg, offset), (0.0, offset)],
                    dtype='d')


def _off_the_line(line, point):
    """How far a point is from a polyline, in metres."""
    line = np.asarray(line, dtype='d')
    starts, ends = line[:-1], line[1:]
    along = ends - starts
    squared = np.maximum((along * along).sum(axis=1), 1e-12)
    fraction = np.clip(((point - starts) * along).sum(axis=1) / squared,
                       0.0, 1.0)
    closest = starts + along * fraction[:, None]
    return float(np.linalg.norm(closest - point, axis=1).min())


def _gentle(points=9, radius=400.0):
    """A wide arc, already easy enough for anything to drive."""
    angles = np.linspace(0.0, np.pi / 2.0, points)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


class TestRoundingACorner:
    def test_nothing_ends_up_tighter_than_it_was_told(self) -> None:
        held = hold_corners(_hairpin(), minimum=25.0)
        assert least_radius(held) >= 25.0 * 0.9

    def test_a_hairpin_is_still_a_hairpin(self) -> None:
        """The road leaves the corner going back the way it came."""
        held = hold_corners(_hairpin(), minimum=25.0)
        out = held[-1] - held[-2]
        back = held[1] - held[0]
        assert float(np.dot(out / np.linalg.norm(out),
                            back / np.linalg.norm(back))) < -0.9

    def test_the_legs_stay_where_they_were_drawn(self) -> None:
        """Which is the whole point: a switchback is drawn to be there."""
        drawn = _hairpin()
        held = hold_corners(drawn, minimum=25.0)
        # The middle of each leg still lies on the line that comes out.
        for middle in ((150.0, 0.0), (300.0, 60.0), (150.0, 120.0)):
            assert _off_the_line(held, np.asarray(middle)) < 1.0

    def test_a_straight_line_is_left_alone(self) -> None:
        straight = np.array([(0.0, 0.0), (100.0, 0.0), (200.0, 0.0)])
        assert np.allclose(hold_corners(straight, minimum=25.0), straight)

    def test_a_corner_already_gentle_enough_is_left_alone(self) -> None:
        gentle = _gentle()
        assert np.allclose(hold_corners(gentle, minimum=25.0), gentle)

    def test_it_puts_points_round_the_corner_rather_than_at_it(self) -> None:
        held = hold_corners(_hairpin(), minimum=25.0)
        assert len(held) > len(_hairpin())

    def test_two_points_are_nothing_to_round(self) -> None:
        line = np.array([(0.0, 0.0), (100.0, 0.0)])
        assert np.allclose(hold_corners(line, minimum=25.0), line)

    def test_no_minimum_leaves_the_plan_as_drawn(self) -> None:
        drawn = _hairpin()
        assert np.allclose(hold_corners(drawn, minimum=0.0), drawn)


class TestNotOverrunningTheLegs:
    def test_a_corner_between_short_legs_still_fits(self) -> None:
        """A fillet longer than the leg it stands on would run into the next."""
        tight = np.array([(0.0, 0.0), (30.0, 0.0), (30.0, 10.0), (0.0, 10.0)])
        held = hold_corners(tight, minimum=100.0)
        assert np.all(np.isfinite(held))
        steps = np.linalg.norm(np.diff(held, axis=0), axis=1)
        assert np.all(steps > 0.0)

    def test_the_line_still_runs_from_where_it_started(self) -> None:
        drawn = _hairpin()
        held = hold_corners(drawn, minimum=25.0)
        assert np.allclose(held[0], drawn[0])
        assert np.allclose(held[-1], drawn[-1])


class TestAgainstOpeningItOut:
    """The two ways to answer a corner too tight to drive, and why a drawn
    switchback wants this one."""

    def _resampled(self, drawn, spacing=6.0):
        """As the generator sees it: a plan is built at the road's spacing, and
        a corner is only a corner once there are points across it."""
        from OpenGLContext.scenegraph.road import resample_polyline
        line = np.stack([drawn[:, 0], np.zeros(len(drawn)), drawn[:, 1]],
                        axis=-1)
        return resample_polyline(line, spacing)[:, [0, 2]]

    def test_relaxing_a_hairpin_moves_it_off_the_line_it_was_drawn_on(self) -> None:
        drawn = self._resampled(_hairpin())
        opened = hold_radius(drawn, minimum=60.0)
        assert np.linalg.norm(opened - drawn, axis=1).max() > 5.0

    def test_rounding_it_stays_beside_every_drawn_point(self) -> None:
        """A fillet cuts the apex, by about its own radius and no more; the
        relaxation moves the whole line instead."""
        drawn = _hairpin()
        held = hold_corners(drawn, minimum=25.0)
        for point in drawn:
            assert _off_the_line(held, point) < 25.0


class TestOnAClosedCircuit:
    def test_the_corner_at_the_join_is_rounded_too(self) -> None:
        square = np.array([(0.0, 0.0), (200.0, 0.0), (200.0, 200.0),
                           (0.0, 200.0)])
        held = hold_corners(square, minimum=30.0, closed=True)
        assert least_radius(held, closed=True) >= 30.0 * 0.9

    def test_it_still_comes_back_to_where_it_started(self) -> None:
        square = np.array([(0.0, 0.0), (200.0, 0.0), (200.0, 200.0),
                           (0.0, 200.0)])
        held = hold_corners(square, minimum=30.0, closed=True)
        assert np.linalg.norm(held[0] - held[-1]) > 1.0    # not duplicated
        assert len(held) > len(square)


class TestThroughTheGenerator:
    def _world(self, plan, **named):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        named.setdefault('extent', 2048.0)
        named.setdefault('resolution', 17)
        named.setdefault('tree_density', 0.0)
        return ProceduralWorld(route=plan, closed=False, **named)

    def test_a_drawn_switchback_survives_being_built(self) -> None:
        plan = _hairpin(leg=400.0, offset=60.0)
        line = self._world(plan).circuit().points
        out = line[-1] - line[-2]
        back = line[1] - line[0]
        assert float(np.dot(out / np.linalg.norm(out),
                            back / np.linalg.norm(back))) < -0.8

    def test_a_hairpin_with_room_gets_the_radius_the_speed_asks_for(self) -> None:
        from OpenGLContext_editor.world.procedural import CIRCUIT_DESIGN_SPEED
        from OpenGLContext_editor.world.route import cornering_radius
        wanted = cornering_radius(CIRCUIT_DESIGN_SPEED)
        plan = _hairpin(leg=1400.0, offset=wanted * 5.0)
        built = self._world(plan).circuit().points[:, [0, 2]]
        assert least_radius(built) >= wanted * 0.9

    def test_a_hairpin_with_less_room_gets_the_best_that_fits(self) -> None:
        """Legs sixty metres apart cannot carry a hundred-and-eighty-metre
        corner; what they can carry is what the road is built with, and it is
        a corner rather than the vertex it replaced."""
        plan = _hairpin(leg=400.0, offset=60.0)
        built = self._world(plan).circuit().points[:, [0, 2]]
        bare = self._world(np.asarray(plan)).circuit()
        assert least_radius(built) > 20.0
        assert least_radius(built) < 60.0

    def test_the_road_still_goes_where_it_was_drawn(self) -> None:
        plan = _hairpin(leg=400.0, offset=60.0)
        built = self._world(plan).circuit().points[:, [0, 2]]
        for point in plan:
            assert float(np.linalg.norm(built - point, axis=1).min()) < 60.0
