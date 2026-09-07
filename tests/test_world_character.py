"""What kind of road this is, here.

A road held to one design speed, one grade limit, one smoothing and one cleared
width is a road of one character: a lap of it is the same lap however long it
is. What is here works out how those four vary along an alignment, and every one
of them is derived from something -- the corner the road is on, the land it is
crossing, how far a driver has to be able to see -- rather than sprinkled about.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import corner_speed

from OpenGLContext_editor.world.character import (
    RoadCharacter,
    corner_radii,
    road_character,
)


def _flat(x, z):
    return np.zeros_like(np.asarray(x, 'd'))


def _hill(x, z):
    """A hillside the road climbs, and a flat run either side of it.

    The road runs towards -z, so the ground has to *rise* as z falls for the
    drive along it to be a climb.
    """
    z = np.asarray(z, 'd')
    return 120.0 / (1.0 + np.exp((z + 900.0) / 90.0))


def _ring(radius=900.0, corners=9):
    turn = np.linspace(0.0, 2.0 * np.pi, corners, endpoint=False)
    return np.stack([radius * np.cos(turn), radius * np.sin(turn)], axis=-1)


def _straight(length=2400.0, count=13):
    z = np.linspace(0.0, -length, count)
    return np.stack([np.zeros(count), z], axis=-1)


class TestTheCornersACircuitGets:
    def _radii(self, **kwargs):
        kwargs.setdefault('seed', 3)
        return corner_radii(_ring(), design_radius=270.0, **kwargs)

    def test_one_per_vertex_of_the_plan(self) -> None:
        assert len(self._radii()) == len(_ring())

    def test_most_of_them_are_the_road_s_own_corner(self) -> None:
        assert np.median(self._radii()) == pytest.approx(270.0)

    def test_none_of_them_is_tighter_than_the_road_allows(self) -> None:
        assert self._radii(tightest=80.0).min() >= 80.0

    def test_at_least_one_is_a_corner_worth_braking_for(self) -> None:
        """A lap of corners all worth the same speed is one corner repeated."""
        assert corner_speed(float(self._radii().min())) * 3.6 < 140.0

    def test_at_least_one_is_faster_than_the_road_was_laid_out_for(self) -> None:
        assert self._radii().max() > 270.0 * 1.5

    def test_asking_for_no_spread_gives_the_road_s_own_corner_throughout(self) -> None:
        assert np.allclose(self._radii(spread=0.0), 270.0)

    def test_the_same_seed_is_the_same_lap(self) -> None:
        assert np.allclose(self._radii(seed=7), self._radii(seed=7))

    def test_a_different_seed_is_a_different_lap(self) -> None:
        assert not np.allclose(self._radii(seed=7), self._radii(seed=8))


class TestHowFastEachStretchIsFor:
    def _character(self, plan=None, **kwargs):
        kwargs.setdefault('spacing', 6.0)
        kwargs.setdefault('closed', True)
        return road_character(plan if plan is not None else _ring(), _flat,
                              design_speed=200.0 / 3.6, **kwargs)

    def test_it_answers_one_figure_per_alignment_point(self) -> None:
        from OpenGLContext_editor.world.road import points_along
        found = self._character()
        wanted = points_along(_ring(), 6.0, closed=True)
        for each in (found.design_speed, found.grade_limit, found.smoothing,
                     found.clearance):
            assert len(each) == wanted

    def test_a_straight_is_for_the_speed_the_road_is_for(self) -> None:
        found = road_character(_straight(), _flat, design_speed=200.0 / 3.6,
                               spacing=6.0, closed=False)
        assert found.design_speed.max() == pytest.approx(200.0 / 3.6)

    def test_a_tight_corner_is_for_the_speed_it_can_be_taken_at(self) -> None:
        """Not the speed of the straight before it: a crest inside a hairpin
        rounded for two hundred is a crest nobody meets at two hundred."""
        from OpenGLContext_editor.world.route import hold_corners
        plan = hold_corners(_ring(), 60.0, closed=True)
        found = road_character(plan, _flat, design_speed=200.0 / 3.6,
                               spacing=6.0, closed=True)
        assert found.design_speed.min() * 3.6 < 120.0

    def test_it_never_asks_for_more_than_the_road_is_for(self) -> None:
        assert self._character().design_speed.max() <= 200.0 / 3.6 + 1e-9


class TestWhereTheRoadIsAllowedToClimb:
    def _character(self, **kwargs):
        kwargs.setdefault('spacing', 6.0)
        return road_character(_straight(), _hill, design_speed=200.0 / 3.6,
                              closed=False, grade_limit=0.06,
                              steep_grade=0.15, **kwargs)

    def test_the_flat_run_is_held_to_the_road_s_own_grade(self) -> None:
        assert self._character().grade_limit[:20].max() == pytest.approx(0.06)

    def test_the_hillside_is_allowed_a_climb(self) -> None:
        """Where the land itself goes up hard, the road goes up with it rather
        than standing off it on an embankment for half a kilometre."""
        assert self._character().grade_limit.max() > 0.10

    def test_it_never_allows_more_than_it_is_told(self) -> None:
        assert self._character().grade_limit.max() <= 0.15 + 1e-9

    def test_ground_that_never_climbs_never_gets_one(self) -> None:
        found = road_character(_straight(), _flat, design_speed=55.0,
                               spacing=6.0, closed=False, grade_limit=0.06,
                               steep_grade=0.15)
        assert np.allclose(found.grade_limit, 0.06)


class TestWhichStretchesAreLeftAsTheyLie:
    def test_a_fast_stretch_is_ironed_out(self) -> None:
        """A bump taken at two hundred is a car in the air; one taken at eighty
        is a road with some character in it."""
        found = road_character(_straight(), _flat, design_speed=200.0 / 3.6,
                               spacing=6.0, closed=False)
        assert found.smoothing.min() > 40.0

    def test_a_slow_stretch_keeps_the_ground_s_own_shape(self) -> None:
        from OpenGLContext_editor.world.route import hold_corners
        plan = hold_corners(_ring(), 60.0, closed=True)
        found = road_character(plan, _flat, design_speed=200.0 / 3.6,
                               spacing=6.0, closed=True)
        assert found.smoothing.min() < 20.0


class TestHowWideTheTreesAreCutBack:
    def _clearance(self, radius, **kwargs):
        from OpenGLContext_editor.world.route import hold_corners
        plan = hold_corners(_ring(), radius, closed=True)
        return road_character(plan, _flat, design_speed=200.0 / 3.6,
                              spacing=6.0, closed=True, **kwargs).clearance

    def test_a_straight_needs_nothing_cut_back_to_see_down_it(self) -> None:
        found = road_character(_straight(), _flat, design_speed=200.0 / 3.6,
                               spacing=6.0, closed=False).clearance
        assert found.max() == pytest.approx(found.min())

    def test_a_corner_near_the_design_radius_is_opened_out(self) -> None:
        """The one a driver most needs to see the exit of: quick enough that
        stopping takes a while, tight enough that the trees are in the way."""
        assert self._clearance(270.0).max() > 9.0

    def test_a_sweeper_needs_nothing_the_road_does_not_already_give(self) -> None:
        """Straight enough to see round from the carriageway itself, so the
        forest comes up to the verge."""
        found = self._clearance(1200.0)
        assert found.max() == pytest.approx(found.min())

    def test_a_hairpin_is_opened_out_too_though_it_is_slow(self) -> None:
        """Slow, but so tight that the trees on the inside are the corner."""
        assert self._clearance(60.0).max() > 9.0

    def test_it_is_never_wider_than_it_is_allowed(self) -> None:
        assert self._clearance(270.0, most_clearing=9.5).max() <= 9.5 + 1e-9

    def test_it_is_never_narrower_than_the_corridor_the_road_needs(self) -> None:
        assert self._clearance(1200.0, clearance=6.9).min() >= 6.9 - 1e-9


class TestWhereTheRoadIsWideEnoughToBePassedOn:
    def _widening(self, ground, **kwargs):
        kwargs.setdefault('climbing_lane', 3.6)
        return road_character(_straight(), ground, design_speed=200.0 / 3.6,
                              spacing=6.0, closed=False, **kwargs).widening

    def test_a_road_told_of_no_lane_is_one_width_throughout(self) -> None:
        assert np.allclose(self._widening(_hill, climbing_lane=0.0), 0.0)

    def test_flat_country_earns_none(self) -> None:
        """Nothing is held up on the level, so nothing needs passing there."""
        assert np.allclose(self._widening(_flat), 0.0)

    def test_a_long_climb_earns_one(self) -> None:
        assert self._widening(_hill).max() == pytest.approx(3.6)

    def test_it_is_only_the_climb_that_gets_it(self) -> None:
        found = self._widening(_hill)
        assert found.min() == pytest.approx(0.0)
        assert float(np.mean(found > 0.1)) < 0.8

    def test_a_descent_earns_none(self) -> None:
        """What is slow going up a hill is not slow coming down it."""
        def downhill(x, z):
            return -_hill(x, z)
        assert np.allclose(self._widening(downhill), 0.0)

    def test_it_opens_out_rather_than_stepping(self) -> None:
        """A lane that began at a vertex would begin in mid-air."""
        found = self._widening(_hill)
        assert np.abs(np.diff(found)).max() < 3.6 * 0.25


class TestACharacterIsAWholeRoad:
    def test_it_reports_what_it_varies(self) -> None:
        found = road_character(_ring(), _flat, design_speed=55.0, spacing=6.0,
                               closed=True)
        assert isinstance(found, RoadCharacter)
        assert set(found.summary()) == {
            'designSpeed', 'gradeLimit', 'smoothing', 'clearance', 'widening'}

    def test_a_road_of_one_character_says_so(self) -> None:
        found = road_character(_straight(), _flat, design_speed=55.0,
                               spacing=6.0, closed=False)
        assert not found.varies()

    def test_a_road_of_several_says_so_too(self) -> None:
        from OpenGLContext_editor.world.route import hold_corners
        found = road_character(hold_corners(_ring(), 60.0, closed=True), _hill,
                               design_speed=55.0, spacing=6.0, closed=True)
        assert found.varies()
