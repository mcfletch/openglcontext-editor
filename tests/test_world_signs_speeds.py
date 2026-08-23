"""What a sign says about speed: the bend's own, and the road's own.

A generated road knows how tightly it turns, and how tightly it turns is how
fast it may be taken -- so the number on the tab under a bend sign is derivable
from the alignment in exactly the way the symbol on the plate above it is. The
posted limit is not derivable, because it is a decision rather than a
measurement, so it is told to the road and then repeated along it.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile, advisory_speed, corner_speed
from OpenGLContext.scenegraph.roadsigns import LIMIT

from OpenGLContext_editor.world.road import RoadPath
from OpenGLContext_editor.world.signs import (
    LIMIT_SPACING,
    sign_placements,
    warn_of,
)

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
SPEED = 42.0


def _bend(radius=60.0, sweep=np.pi / 2, run=300.0, count=241):
    """A straight, then a constant-radius bend of ``radius``, turning left."""
    lead = np.linspace(0.0, run, count // 2)
    line = [(0.0, 0.0, z) for z in lead]
    angle = np.linspace(0.0, sweep, count - count // 2)
    for a in angle[1:]:
        line.append((-radius * (1 - np.cos(a)), 0.0, run + radius * np.sin(a)))
    return np.asarray(line)


def _straight(length=6000.0, count=1001):
    z = np.linspace(0.0, length, count)
    return np.stack([np.zeros(count), np.zeros(count), z], axis=-1)


def _path(points, ops=None):
    return RoadPath(points, profile=PROFILE, ops=ops)


class TestWhatABendIsWorth:
    def test_a_bend_sign_carries_a_speed(self) -> None:
        found = warn_of(_path(_bend(radius=45.0)), SPEED)
        assert found and found[0].speed > 0

    def test_and_it_is_the_speed_that_bend_allows(self) -> None:
        found = warn_of(_path(_bend(radius=45.0)), SPEED)
        assert found[0].speed == pytest.approx(advisory_speed(45.0), abs=10.0)

    def test_a_tighter_bend_says_a_lower_number(self) -> None:
        tight = warn_of(_path(_bend(radius=30.0)), SPEED)
        open_ = warn_of(_path(_bend(radius=70.0)), SPEED)
        assert tight[0].speed < open_[0].speed

    def test_the_number_is_well_inside_what_the_bend_holds(self) -> None:
        """A sign carrying the limit is a sign that is wrong for a wet road."""
        found = warn_of(_path(_bend(radius=45.0)), SPEED)
        assert found[0].speed < corner_speed(45.0) * 3.6

    def test_a_sign_that_is_not_about_a_bend_carries_no_speed(self) -> None:
        found = [one for one in warn_of(_path(_bend(radius=45.0)), SPEED)
                 if one.kind not in ('bend-left', 'bend-right', 'double-bend')]
        assert all(one.speed == 0 for one in found)


class TestWhatTheRoadIsWorth:
    def test_a_long_road_is_posted(self) -> None:
        found = warn_of(_path(_straight()), SPEED, limit=100)
        assert [one for one in found if one.kind == LIMIT]

    def test_and_the_sign_says_the_limit_it_was_given(self) -> None:
        found = warn_of(_path(_straight()), SPEED, limit=80)
        assert all(one.speed == 80 for one in found if one.kind == LIMIT)

    def test_they_are_repeated_along_it(self) -> None:
        found = [one.station for one in warn_of(_path(_straight()), SPEED,
                                                limit=100)
                 if one.kind == LIMIT]
        assert len(found) >= 2
        assert np.diff(found).max() <= LIMIT_SPACING * 1.5

    def test_a_road_that_is_not_posted_carries_none(self) -> None:
        found = warn_of(_path(_straight()), SPEED, limit=0)
        assert not [one for one in found if one.kind == LIMIT]

    def test_and_it_is_still_a_road_posted_at_intervals(self) -> None:
        found = warn_of(_path(_straight(length=9000.0, count=1501)), SPEED,
                        limit=100)
        limits = [one.station for one in found if one.kind == LIMIT]
        assert len(limits) >= 5

    def test_a_limit_does_not_stand_on_top_of_a_warning(self) -> None:
        """Two signs twenty metres apart are two signs nobody reads."""
        found = warn_of(_path(_bend(radius=40.0)), SPEED, limit=100)
        limits = [one.station for one in found if one.kind == LIMIT]
        warnings = [one.station for one in found if one.kind != LIMIT]
        for at in limits:
            assert all(abs(at - other) > 100.0 for other in warnings)


class TestARoadWithSignsAllOverIt:
    """The shipped circuit: a bend warning every few hundred metres.

    A limit sign that gave way to every warning near it is a road posted
    nowhere, which is what a 4 km circuit with a warning at 1518 m and another
    at 2997 m produced. A limit sign is not *about* a place, so it moves.
    """

    @pytest.fixture(scope='class')
    def circuit(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=2048.0, seed=11).circuit()

    def test_it_is_posted(self, circuit) -> None:
        found = warn_of(circuit, SPEED, limit=100)
        assert [one for one in found if one.kind == LIMIT]

    def test_about_as_often_as_the_interval_says(self, circuit) -> None:
        found = warn_of(circuit, SPEED, limit=100)
        limits = [one for one in found if one.kind == LIMIT]
        assert len(limits) >= int(circuit.stations[-1] // LIMIT_SPACING)

    def test_and_none_of_them_shares_a_place_with_a_warning(self, circuit
                                                            ) -> None:
        found = warn_of(circuit, SPEED, limit=100)
        limits = [one.station for one in found if one.kind == LIMIT]
        warnings = [one.station for one in found if one.kind != LIMIT]
        for at in limits:
            assert all(abs(at - other) > 100.0 for other in warnings)


class TestWhatReachesTheWorld:
    def test_a_placement_carries_the_speed_its_sign_says(self) -> None:
        path = _path(_bend(radius=40.0))
        warnings = warn_of(path, SPEED)
        placed = sign_placements(path, warnings)
        assert [one for one in placed if one.speed > 0]

    def test_and_the_face_it_is_built_from(self) -> None:
        path = _path(_bend(radius=40.0))
        placed = sign_placements(path, warn_of(path, SPEED))
        found = placed[0]
        assert found.face.kind == found.kind
        assert found.face.speed == found.speed


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
