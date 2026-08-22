"""A road that is not the same road all the way round.

An alignment held to one design speed, one grade limit and one smoothing from
end to end is a road of one character, and a lap of it is the same lap however
long it is. What is here is the machinery that lets the limits *vary* along the
road -- a hairpin where the land wants one, a climb steeper than the rest, a
stretch that follows the ground instead of ironing it out -- and the rule that
each of those is still a limit the whole way.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.road import follow_terrain


def _flat(x, z):
    return np.zeros_like(np.asarray(x, 'd'))


def _ridge(x, z):
    """Ground that climbs steadily and carries a metre-scale ripple."""
    x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
    return 0.09 * -z + 1.2 * np.sin(z * 0.35)


def _plan(length=600.0, count=61):
    z = np.linspace(0.0, -length, count)
    return np.stack([np.zeros(count), z], axis=-1)


def _grades(line):
    steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
    return np.abs(np.diff(line[:, 1])) / np.maximum(steps, 1e-9)


class TestAGradeLimitThatVaries:
    def _followed(self, maximum):
        return follow_terrain(_plan(), _ridge, spacing=5.0, smoothing=0.0,
                              maximum_grade=maximum)

    def test_one_figure_still_holds_the_whole_road(self) -> None:
        assert _grades(self._followed(0.05)).max() <= 0.05 + 1e-6

    def test_a_figure_per_point_holds_each_stretch_to_its_own(self) -> None:
        line = self._followed(_flat(np.zeros(121), 0) + 0.04)
        assert _grades(line).max() <= 0.04 + 1e-6

    def test_a_stretch_allowed_more_climbs_more(self) -> None:
        """The steep section is the second quarter of the road."""
        limit = np.full(121, 0.03)
        limit[30:60] = 0.14
        line = self._followed(limit)
        grades = _grades(line)
        assert grades[31:58].max() > 0.10
        assert grades[:28].max() <= 0.03 + 1e-6
        assert grades[62:].max() <= 0.03 + 1e-6

    def test_a_limit_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match='grade'):
            self._followed(np.full(7, 0.05))


class TestADesignSpeedThatVaries:
    def _crest(self, speed):
        """A road over a ridge, which is where a design speed shows."""
        z = np.linspace(0.0, -600.0, 121)
        plan = np.stack([np.zeros(121), z], axis=-1)

        def ridge(x, zz):
            return 40.0 * np.exp(-((np.asarray(zz, 'd') + 300.0) / 70.0) ** 2)
        return follow_terrain(plan, ridge, spacing=5.0, smoothing=0.0,
                              maximum_grade=0.12, design_speed=speed)

    def _curvature(self, line):
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        grade = np.diff(line[:, 1]) / np.maximum(steps, 1e-9)
        return np.abs(np.diff(grade)) / np.maximum(steps[:-1], 1e-9)

    def test_one_speed_still_rounds_the_whole_road(self) -> None:
        fast = self._curvature(self._crest(55.0)).max()
        slow = self._curvature(self._crest(20.0)).max()
        assert fast < slow

    def test_a_speed_per_point_rounds_each_stretch_for_its_own(self) -> None:
        """Slow where the crest is, fast elsewhere: the crest keeps a corner a
        fast road would have had spread into a vertical curve."""
        slow = np.full(121, 55.0)
        slow[45:75] = 18.0
        varied = self._curvature(self._crest(slow)).max()
        assert varied > self._curvature(self._crest(55.0)).max()

    def test_a_speed_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match='design speed'):
            self._crest(np.full(9, 40.0))


class TestSmoothingThatVaries:
    def _followed(self, smoothing):
        return follow_terrain(_plan(), _ridge, spacing=5.0,
                              smoothing=smoothing, maximum_grade=0.5)

    def _ripple(self, line, first, last):
        """How much of the ground's own ripple survived, over a stretch."""
        return float(np.std(np.diff(np.diff(line[first:last, 1]))))

    def test_one_window_still_smooths_the_whole_road(self) -> None:
        assert self._ripple(self._followed(0.0), 10, 110) \
            > self._ripple(self._followed(60.0), 10, 110)

    def test_a_window_per_point_leaves_the_bumps_where_they_belong(self) -> None:
        window = np.full(121, 60.0)
        window[30:70] = 0.0                      # a stretch left as it lies
        line = self._followed(window)
        assert self._ripple(line, 35, 65) > 5.0 * self._ripple(line, 85, 115)

    def test_a_window_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match='smoothing'):
            self._followed(np.full(5, 20.0))

    def test_a_varying_window_matches_a_fixed_one_where_it_is_fixed(self) -> None:
        """The variable-width filter is the same filter, so a window that does
        not vary gives what the fixed one gave."""
        fixed = self._followed(40.0)
        varied = self._followed(np.full(121, 40.0))
        assert np.allclose(fixed[:, 1], varied[:, 1], atol=1e-9)


class TestCornersThatDifferFromEachOther:
    def _square(self, side=800.0):
        """Four corners, so each one's radius can be asked for separately."""
        return np.array([(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)])

    def _radii(self, held, plan):
        """The tightest radius the built line reaches near each drawn corner."""
        from OpenGLContext_editor.world.route import _neighbours, _radius
        before, after = _neighbours(held, True)
        radius = _radius(before, held, after)
        found = []
        for corner in plan:
            near = np.linalg.norm(held - corner, axis=1) < 400.0
            found.append(float(radius[near].min()))
        return found

    def test_one_radius_still_holds_every_corner(self) -> None:
        from OpenGLContext_editor.world.route import hold_corners
        plan = self._square()
        found = self._radii(hold_corners(plan, 120.0, closed=True), plan)
        assert np.allclose(found, 120.0, rtol=0.05)

    def test_a_radius_per_corner_gives_each_one_its_own(self) -> None:
        from OpenGLContext_editor.world.route import hold_corners
        plan = self._square()
        wanted = np.array([40.0, 120.0, 240.0, 120.0])
        found = self._radii(hold_corners(plan, wanted, closed=True), plan)
        assert np.allclose(found, wanted, rtol=0.08)

    def test_a_hairpin_is_tighter_than_the_road_it_is_on(self) -> None:
        from OpenGLContext.scenegraph.road import corner_speed
        from OpenGLContext_editor.world.route import hold_corners
        plan = self._square()
        wanted = np.array([45.0, 260.0, 260.0, 260.0])
        found = self._radii(hold_corners(plan, wanted, closed=True), plan)
        assert corner_speed(found[0]) * 3.6 < 90.0
        assert min(corner_speed(r) for r in found[1:]) * 3.6 > 180.0

    def test_a_radius_of_the_wrong_length_is_refused(self) -> None:
        from OpenGLContext_editor.world.route import hold_corners
        with pytest.raises(ValueError, match='radi'):
            hold_corners(self._square(), np.array([40.0, 120.0]), closed=True)


class TestACircuitWithSomeVarietyInIt:
    def _plan(self, **kwargs):
        from OpenGLContext_editor.world.procedural import circuit_plan
        return circuit_plan(1300.0, 1000.0, **kwargs)

    def _legs(self, plan):
        return np.linalg.norm(np.diff(np.vstack([plan, plan[:1]]), axis=0),
                              axis=1)

    def _turns(self, plan):
        into = plan - np.roll(plan, 1, axis=0)
        out = np.roll(plan, -1, axis=0) - plan
        into = into / np.linalg.norm(into, axis=1, keepdims=True)
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', into, out),
                                            -1.0, 1.0)))

    def test_it_still_closes_into_a_circuit(self) -> None:
        plan = self._plan(variation=0.8, seed=3)
        assert len(plan) > 4
        assert self._legs(plan).min() > 50.0

    def test_asking_for_none_gives_the_even_circuit_it_always_gave(self) -> None:
        assert np.allclose(self._plan(variation=0.0), self._plan())

    def test_its_legs_are_not_all_the_same_length(self) -> None:
        """A circuit of equal legs has one straight repeated; what makes a lap
        worth learning is a long one somewhere and a short one elsewhere."""
        legs = self._legs(self._plan(variation=0.8, seed=3))
        assert legs.max() > 2.2 * legs.min()

    def test_its_corners_do_not_all_turn_through_the_same_angle(self) -> None:
        turns = self._turns(self._plan(variation=0.8, seed=3))
        assert turns.max() - turns.min() > 30.0

    def test_it_is_the_same_circuit_for_the_same_seed(self) -> None:
        assert np.allclose(self._plan(variation=0.8, seed=5),
                           self._plan(variation=0.8, seed=5))

    def test_a_different_seed_is_a_different_circuit(self) -> None:
        assert not np.allclose(self._plan(variation=0.8, seed=5),
                               self._plan(variation=0.8, seed=6))


class TestACircuitThatIsNotTheSameRoadAllTheWayRound:
    """The whole of it, through the generator: a lap with a corner to brake
    for, a climb, a stretch left rough and a clearing where one is needed."""

    def _world(self, variety):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=4096.0, structures=False,
                               variety=variety)

    def _corners(self, path):
        from OpenGLContext.scenegraph.road import plan_curvature
        turning = np.abs(plan_curvature(path.points, closed=True))
        return np.where(turning > 1e-9, 1.0 / np.maximum(turning, 1e-12),
                        np.inf)

    def _grades(self, path):
        steps = np.linalg.norm(np.diff(path.points[:, [0, 2]], axis=0), axis=1)
        return np.abs(np.diff(path.points[:, 1])) / np.maximum(steps, 1e-9)

    def test_asking_for_none_still_builds_the_road_one_figure_gives(self) -> None:
        """Nothing tighter than the corner the road was laid out to, and one
        cleared width from end to end -- the road this generator built before
        any of its stretches had a character of their own."""
        world = self._world(0.0)
        path = world.circuit()
        assert self._corners(path).min() == pytest.approx(
            world.corner_radius(), rel=0.02)
        assert float(np.ptp(path.clearance)) < 1e-9

    def test_variety_puts_a_corner_on_it_the_road_was_not_laid_out_for(self) -> None:
        world = self._world(1.0)
        assert self._corners(world.circuit()).min() < world.corner_radius() * 0.5

    def test_it_gets_a_corner_worth_braking_for(self) -> None:
        from OpenGLContext.scenegraph.road import corner_speed
        path = self._world(1.0).circuit()
        held = [corner_speed(float(r), bank=float(b)) * 3.6
                for r, b in zip(self._corners(path), path.bank, strict=True)]
        assert min(held) < 130.0

    def test_it_still_has_somewhere_to_go_quickly(self) -> None:
        """A lap that is all slow corners is a lap nobody passes on."""
        from OpenGLContext.scenegraph.road import corner_speed
        path = self._world(1.0).circuit()
        held = np.array([corner_speed(float(r), bank=float(b)) * 3.6
                         for r, b in zip(self._corners(path), path.bank,
                                         strict=True)])
        assert float(np.mean(held > 190.0)) > 0.4

    def test_it_climbs_harder_where_the_land_does(self) -> None:
        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_MAX_GRADE, CIRCUIT_STEEP_GRADE,
        )
        found = self._grades(self._world(1.0).circuit()).max()
        assert found > CIRCUIT_MAX_GRADE * 1.15
        assert found <= CIRCUIT_STEEP_GRADE + 1e-6

    def test_it_is_opened_out_where_a_driver_has_to_see_round(self) -> None:
        path = self._world(1.0).circuit()
        assert path.clearance.max() > path.clearance.min() + 3.0

    def test_no_tree_stands_inside_the_clearing_it_is_beside(self) -> None:
        """The widened corridor is the one the forest is actually kept out of,
        not a number the road carries and nothing reads."""
        world = self._world(1.0)
        placed = world.scatter().positions
        path = world.circuit()
        found = path.sample(placed[:, 0], placed[:, 2], radius=40.0)
        beside = path.clearance_along()[found.segment]
        close = np.isfinite(found.distance)
        assert np.all(found.distance[close] >= beside[close] - 1e-6)

    def test_the_same_seed_is_the_same_circuit(self) -> None:
        assert np.allclose(self._world(1.0).circuit().points,
                           self._world(1.0).circuit().points)
