"""Where a circuit's start/finish gantry stands, worked out from the road.

A lap begins where the centreline does, so nothing here is authored: the
placement takes the crown at that station, spans the running surface, and turns
the gantry to face along the road. What it has to get right is the ground -- the
two legs stand off either side of the road and the land under them is rarely
level with the tarmac, so each is given its own drop.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.gantry import GantryProfile
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.world.gantry import FOOTING, StartFinish, start_finish
from OpenGLContext_editor.world.road import RoadPath

PROFILE = RoadProfile(lane_width=3.6, lanes=2, shoulder_width=1.5)


def _straight(length=1000.0, count=101, height=0.0):
    z = np.linspace(0.0, length, count)
    return np.stack([np.zeros(count), np.full(count, height), z], axis=-1)


def _path(points):
    return RoadPath(points, profile=PROFILE)


def _turned(points):
    """The same road running along +X instead of +Z."""
    return np.stack([points[:, 2], points[:, 1], points[:, 0]], axis=-1)


class TestWhereItStands:
    def test_it_is_at_the_beginning_of_the_lap(self) -> None:
        found = start_finish(_path(_straight()))
        assert pytest.approx([0.0, 0.0, 0.0], abs=1e-6) == list(found.position)

    def test_a_caller_can_put_it_further_along(self) -> None:
        found = start_finish(_path(_straight()), station=250.0)
        assert pytest.approx(250.0, abs=10.0) == float(found.position[2])

    def test_it_stands_at_the_height_of_the_road(self) -> None:
        found = start_finish(_path(_straight(height=37.0)))
        assert pytest.approx(37.0, abs=1e-6) == float(found.position[1])

    def test_it_is_a_start_finish(self) -> None:
        assert isinstance(start_finish(_path(_straight())), StartFinish)


class TestHowWideItIs:
    def test_the_line_reaches_across_the_carriageway(self) -> None:
        found = start_finish(_path(_straight()))
        assert pytest.approx(PROFILE.carriageway_width,
                             abs=1e-6) == found.width

    def test_the_legs_stand_clear_of_the_running_surface(self) -> None:
        profile = GantryProfile()
        found = start_finish(_path(_straight()), profile=profile)
        clear = PROFILE.carriageway_width / 2.0 + PROFILE.shoulder_width
        assert found.span / 2.0 >= clear + profile.margin - 1e-6

    def test_they_do_not_stand_out_in_the_trees(self) -> None:
        """Inside the corridor the road already keeps clear of vegetation."""
        found = start_finish(_path(_straight()))
        assert found.span <= PROFILE.total_width

    def test_the_line_is_painted_on_the_road_s_own_camber(self) -> None:
        found = start_finish(_path(_straight()))
        assert found.crossfall == PROFILE.crossfall


class TestWhichWayItFaces:
    def test_a_road_running_north_turns_it_not_at_all(self) -> None:
        found = start_finish(_path(_straight()))
        assert pytest.approx(0.0, abs=1e-6) == found.yaw

    def test_a_road_running_east_turns_it_a_quarter(self) -> None:
        found = start_finish(_path(_turned(_straight())))
        assert pytest.approx(np.pi / 2.0, abs=1e-6) == abs(found.yaw)

    def test_the_beam_crosses_the_road_rather_than_lying_along_it(self) -> None:
        """The prototype spans X with the road along -Z, so the turned beam has
        to come out square to the direction of travel."""
        for line in (_straight(), _turned(_straight())):
            found = start_finish(_path(line))
            beam = np.array([np.cos(found.yaw), 0.0, -np.sin(found.yaw)])
            forward = line[1] - line[0]
            forward = forward / np.linalg.norm(forward)
            assert pytest.approx(0.0, abs=1e-6) == float(beam @ forward)


class TestTheGroundUnderTheLegs:
    def test_level_ground_still_sinks_the_feet_a_little(self) -> None:
        """A coarse terrain tile is not the surface the drop was measured on;
        a foot resting exactly on it shows daylight underneath."""
        found = start_finish(_path(_straight()), ground=lambda x, z: 0.0)
        assert pytest.approx((FOOTING, FOOTING), abs=1e-6) == found.drops

    def test_a_leg_over_a_drop_off_reaches_down_to_it(self) -> None:
        found = start_finish(_path(_straight()),
                             ground=lambda x, z: np.where(np.asarray(x) > 0.0,
                                                          -4.0, 0.0))
        assert pytest.approx(FOOTING, abs=1e-6) == found.drops[0]
        assert pytest.approx(4.0 + FOOTING, abs=1e-6) == found.drops[1]

    def test_ground_above_the_road_does_not_lift_the_gantry(self) -> None:
        """A leg cut into a bank starts at the road, not up the hillside."""
        found = start_finish(_path(_straight()), ground=lambda x, z: 6.0)
        assert pytest.approx((FOOTING, FOOTING), abs=1e-6) == found.drops

    def test_without_a_terrain_the_feet_sit_at_the_road(self) -> None:
        found = start_finish(_path(_straight()))
        assert pytest.approx((FOOTING, FOOTING), abs=1e-6) == found.drops


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestTheShippedWorld:
    """The example world marks its circuit, and leaves its legs room."""

    def _world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=1024.0, resolution=17,
                               field_resolution=129, control_size=256,
                               tree_density=0.0, forest='tiles',
                               ground='tiles')

    def test_the_circuit_is_marked(self) -> None:
        from OpenGLContext_editor.bake.gantry import GantryLayer
        assert any(isinstance(layer, GantryLayer)
                   for layer in self._world().layers())

    def test_a_world_with_no_road_has_no_marker(self) -> None:
        from OpenGLContext_editor.bake.gantry import GantryLayer
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=1024.0, resolution=17, road=False,
                                tree_density=0.0, forest='tiles',
                                ground='tiles')
        assert not any(isinstance(layer, GantryLayer)
                       for layer in world.layers())

    def test_the_gantry_stands_on_the_circuit(self) -> None:
        world = self._world()
        line = world.start_line()
        found = world.circuit().sample(np.array([line.position[0]]),
                                       np.array([line.position[2]]),
                                       radius=50.0)
        assert float(found.distance[0]) < 1.0

    def test_no_boulder_stands_inside_a_leg(self) -> None:
        from OpenGLContext_editor.world.procedural import GANTRY_CLEARANCE
        world = self._world()
        line = world.start_line()
        beam = np.array([np.cos(line.yaw), 0.0, -np.sin(line.yaw)])
        at = np.asarray(line.position, dtype='d')
        stones = world.rocks().positions
        assert len(stones)
        for side in (-1.0, 1.0):
            foot = at + beam * (side * line.span / 2.0)
            away = np.asarray(stones)[:, [0, 2]] - foot[[0, 2]]
            assert float(np.linalg.norm(away, axis=1).min()) >= GANTRY_CLEARANCE
