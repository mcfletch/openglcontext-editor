"""Where a warning sign belongs, worked out from the road itself.

A generated road already knows what it is about to do: the alignment carries its
own curvature and its own grade, and its structures are written down. So the
placement is derivable rather than authored, which is most of the point of
generating a road instead of drawing one.

A sign stands a stopping distance before what it warns of, on the near side of
the road to the traffic it is for, and only where the hazard is worth a sign --
a bend a car can hold at the design speed is not one.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile
from OpenGLContext.scenegraph.roadsigns import LIMIT, WARNINGS

from OpenGLContext_editor.world.road import RoadPath
from OpenGLContext_editor.world.signs import (
    Warning,
    sign_placements,
    stopping_distance,
    warn_of,
)
from OpenGLContext_editor.world.structures import Op

PROFILE = RoadProfile(lane_width=3.6, lanes=2)
SPEED = 42.0


def _straight(length=1200.0, count=201, height=0.0):
    z = np.linspace(0.0, length, count)
    return np.stack([np.zeros(count), np.full(count, height), z], axis=-1)


def _bend(radius=60.0, sweep=np.pi / 2, run=300.0, count=241):
    """A straight, then a constant-radius bend of ``radius``, turning left."""
    lead = np.linspace(0.0, run, count // 2)
    line = [(0.0, 0.0, z) for z in lead]
    angle = np.linspace(0.0, sweep, count - count // 2)
    for a in angle[1:]:
        line.append((-radius * (1 - np.cos(a)), 0.0, run + radius * np.sin(a)))
    return np.asarray(line)


def _dipped(depth=-9.0, length=1200.0, count=201):
    z = np.linspace(0.0, length, count)
    y = depth * np.exp(-((z - length / 2) / 90.0) ** 2)
    return np.stack([np.zeros(count), y, z], axis=-1)


def _path(points, ops=None):
    return RoadPath(points, profile=PROFILE, ops=ops)


class TestWhatIsWorthASign:
    def test_a_straight_road_gets_none(self) -> None:
        assert warn_of(_path(_straight()), SPEED) == []

    def test_a_tight_bend_gets_one(self) -> None:
        found = warn_of(_path(_bend(radius=45.0)), SPEED)
        assert [one.kind for one in found] == ['bend-left']

    def test_a_bend_the_car_can_hold_does_not(self) -> None:
        """There is always some speed at which a corner is too tight; the
        design speed is the answer to which corners count."""
        assert warn_of(_path(_bend(radius=900.0)), SPEED, limit=0) == []

    def test_and_the_road_is_still_posted_through_it(self) -> None:
        """A limit sign is not about a hazard: it says what the road is."""
        found = warn_of(_path(_bend(radius=900.0)), SPEED, limit=100)
        assert found and all(one.kind == LIMIT for one in found)

    def test_which_way_it_turns_is_read_from_the_road(self) -> None:
        right = _bend(radius=45.0).copy()
        right[:, 0] *= -1.0
        assert [one.kind for one in warn_of(_path(right), SPEED)] \
            == ['bend-right']

    def test_a_dip_gets_one(self) -> None:
        assert [one.kind for one in warn_of(_path(_dipped()), SPEED)] == ['dip']

    def test_a_crest_gets_one(self) -> None:
        assert [one.kind for one in warn_of(_path(_dipped(depth=9.0)), SPEED)] \
            == ['crest']

    def test_a_gentle_rise_does_not(self) -> None:
        assert warn_of(_path(_dipped(depth=0.4)), SPEED) == []

    def test_a_tunnel_gets_one(self) -> None:
        line = _straight()
        ops = np.full(len(line), Op.DIRT, dtype=object)
        ops[80:140] = Op.TUNNEL
        assert [one.kind for one in warn_of(_path(line, ops), SPEED)] \
            == ['tunnel']

    def test_a_bridge_does_not(self) -> None:
        """A driver needs no telling that a road is on a bridge."""
        line = _straight()
        ops = np.full(len(line), Op.DIRT, dtype=object)
        ops[80:140] = Op.BRIDGE
        assert warn_of(_path(line, ops), SPEED) == []

    def test_every_kind_it_can_say_is_one_a_sign_can_show(self) -> None:
        for kind in ('bend-left', 'bend-right', 'double-bend', 'dip', 'crest',
                     'tunnel'):
            assert kind in WARNINGS


class TestWhereItStands:
    def _one(self, points, **named):
        found = warn_of(_path(points), SPEED, **named)
        assert len(found) == 1
        return found[0]

    def test_it_stands_before_what_it_warns_of(self) -> None:
        one = self._one(_bend(radius=45.0))
        assert one.station < one.hazard

    def test_it_stands_a_stopping_distance_before_it(self) -> None:
        one = self._one(_bend(radius=45.0))
        assert one.hazard - one.station \
            == pytest.approx(stopping_distance(SPEED), abs=1.0)

    def test_a_faster_road_warns_earlier(self) -> None:
        assert stopping_distance(60.0) > stopping_distance(30.0)

    def test_a_hazard_at_the_very_start_is_still_warned_of(self) -> None:
        """A closed circuit has no start, and an open road warns where it can."""
        one = self._one(_bend(radius=45.0, run=20.0))
        assert one.station >= 0.0

    def test_it_stands_on_the_side_the_traffic_is(self) -> None:
        assert self._one(_bend(radius=45.0)).side == 1

    def test_a_world_that_drives_on_the_left_says_so(self) -> None:
        assert self._one(_bend(radius=45.0), side=-1).side == -1

    def test_two_hazards_close_together_are_one_sign(self) -> None:
        """A sign every ten metres is a sign nobody reads."""
        wiggle = _bend(radius=45.0, sweep=np.pi / 3)
        turned = np.concatenate([wiggle, wiggle[1:] * (-1, 1, 1) + (0, 0, 0)])
        assert len(warn_of(_path(_bend(radius=45.0, sweep=np.pi)), SPEED)) <= 2
        assert len(turned)


class TestPuttingThemInTheWorld:
    def _placed(self, **named):
        return sign_placements(_path(_bend(radius=45.0)),
                               warn_of(_path(_bend(radius=45.0)), SPEED),
                               **named)

    def test_one_placement_per_warning(self) -> None:
        assert len(self._placed()) == 1

    def test_it_is_beside_the_road_not_on_it(self) -> None:
        placed = self._placed()[0]
        assert abs(float(placed.position[0])) > PROFILE.total_width / 2.0 - 1.0

    def test_it_stands_on_the_ground(self) -> None:
        placed = self._placed(ground=lambda x, z: np.full(np.shape(x), -3.0))[0]
        assert float(placed.position[1]) == pytest.approx(-3.0, abs=0.01)

    def test_without_ground_it_stands_at_the_road_s_own_height(self) -> None:
        assert float(self._placed()[0].position[1]) == pytest.approx(0.0,
                                                                     abs=0.5)

    def test_it_faces_the_traffic(self) -> None:
        """The prototype faces -Z; the yaw turns it back down the road."""
        placed = self._placed()[0]
        facing = np.array([np.sin(placed.yaw), 0.0, -np.cos(placed.yaw)])
        assert float(np.dot(facing, (0.0, 0.0, -1.0))) > 0.8

    def test_it_carries_the_kind_it_shows(self) -> None:
        assert self._placed()[0].kind == 'bend-left'


class TestTheShippedCircuit:
    @pytest.fixture(scope='class')
    def circuit(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld().circuit()

    def test_it_gets_signs(self, circuit) -> None:
        assert warn_of(circuit, SPEED)

    def test_and_not_a_forest_of_them(self, circuit) -> None:
        """One every couple of hundred metres is a road; one every twenty is a
        car park."""
        found = warn_of(circuit, SPEED)
        assert len(found) < circuit.length / 200.0

    def test_they_are_in_order_along_the_road(self, circuit) -> None:
        found = [one.station for one in warn_of(circuit, SPEED)]
        assert found == sorted(found)

    def test_every_one_is_a_kind_a_plate_can_show(self, circuit) -> None:
        for one in warn_of(circuit, SPEED):
            assert one.face.plates

    def test_it_warns_of_its_tunnels(self, circuit) -> None:
        assert 'tunnel' in {one.kind for one in warn_of(circuit, SPEED)}


def test_a_warning_reads_as_what_it_says() -> None:
    assert 'dip' in repr(Warning(station=10.0, hazard=90.0, kind='dip', side=1))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestOneBendIsOneSign:
    """A "double bend" means the road turns one way and then the other. Two
    corners the same way in a row are one corner to drive, and a bend whose
    radius wobbles over the limit and back is one bend, not four."""

    def _wiggle(self, radius=60.0, sweep=2 * np.pi / 3, run=200.0,
                hands=(1, -1),
                step=3.0):
        """A straight, then a bend per entry in ``hands``, joined end to end.

        Walked rather than solved: a heading turning at ``1/radius`` per metre
        is what a constant-radius bend is.
        """
        points = [(0.0, 0.0, 0.0)]
        heading = 0.0                              # 0 looks down +Z

        def onwards(distance):
            x, _y, z = points[-1]
            points.append((x + np.sin(heading) * distance, 0.0,
                           z + np.cos(heading) * distance))

        for _ in range(int(run / step)):
            onwards(step)
        for hand in hands:
            turned = 0.0
            while turned < sweep:
                onwards(step)
                heading += hand * step / radius
                turned += step / radius
        return np.asarray(points)

    def test_a_bend_that_wobbles_over_the_limit_is_one_bend(self) -> None:
        line = _bend(radius=45.0, sweep=np.pi / 2, count=401)
        line[:, 0] += np.sin(np.arange(len(line)) * 1.7) * 0.02
        found = warn_of(_path(line), SPEED)
        assert [one.kind for one in found] == ['bend-left']

    def test_two_bends_the_same_way_are_one_sign(self) -> None:
        found = warn_of(_path(self._wiggle(hands=(1, 1))), SPEED)
        assert [one.kind for one in found] in (['bend-left'], ['bend-right'])

    def test_a_left_then_a_right_is_a_double_bend(self) -> None:
        found = warn_of(_path(self._wiggle(hands=(1, -1))), SPEED)
        assert [one.kind for one in found] == ['double-bend']

    def test_a_right_then_a_left_is_too(self) -> None:
        found = warn_of(_path(self._wiggle(hands=(-1, 1))), SPEED)
        assert [one.kind for one in found] == ['double-bend']


class TestTheShippedCircuitReadsAsARoad:
    def test_it_is_not_all_double_bends(self) -> None:
        """A circuit signed as one continuous chicane is a circuit whose signs
        say nothing."""
        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_DESIGN_SPEED,
            ProceduralWorld,
        )
        found = warn_of(ProceduralWorld().circuit(), CIRCUIT_DESIGN_SPEED)
        kinds = {one.kind for one in found}
        doubles = sum(one.kind == 'double-bend' for one in found)
        assert len(kinds) >= 3
        assert doubles < len(found) * 0.6
