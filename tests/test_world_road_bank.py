"""Superelevation: a corner that leans, and the world it leans in.

The engine works out *how much* a corner leans and sweeps the surface with it
(``OpenGLContext.scenegraph.road``). What is here is what a world does about
it: the alignment carries its own lean, the ground beside a banked road meets
the verge it actually has rather than a level one, and the lean travels into
the baked world so that a game builds the same road the bake drew.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile, corner_speed

from OpenGLContext_editor.world.road import RoadPath, conform_terrain


def _flat(x, z):
    return np.zeros_like(np.asarray(x, 'd'))


def _straight(length=200.0, count=21, height=0.0):
    z = np.linspace(0.0, -length, count)
    return np.stack([np.zeros(count), np.full(count, height), z], axis=-1)


class TestARoadThatCarriesItsLean:
    def test_a_road_told_nothing_does_not_lean(self) -> None:
        path = RoadPath(_straight())
        assert np.allclose(path.bank, 0.0)

    def test_a_lean_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match='leans'):
            RoadPath(_straight(count=21), bank=np.zeros(5))

    def test_it_reads_its_lean_off_by_distance_along(self) -> None:
        """A tile carries the road at its own spacing; the lean was worked out
        on the full one, and matching them up is a lookup by distance."""
        lean = np.linspace(0.0, 0.2, 21)
        path = RoadPath(_straight(200.0, count=21), bank=lean)
        assert path.bank_at(np.array([0.0, 100.0, 200.0])) == pytest.approx(
            [0.0, 0.1, 0.2], abs=1e-9)


class TestWhichSideOfTheRoadAPointIsOn:
    def test_a_point_to_the_road_s_right_is_positive(self) -> None:
        """The line runs towards -z, so the road's right hand is towards +x."""
        path = RoadPath(_straight(200.0))
        found = path.sample(np.array([-10.0, 10.0]), np.array([-100.0, -100.0]))
        assert found.side[0] < 0 and found.side[1] > 0

    def test_a_point_on_the_centreline_takes_a_side_rather_than_nothing(self) -> None:
        path = RoadPath(_straight(200.0))
        assert abs(float(path.sample(np.array([0.0]),
                                     np.array([-100.0])).side[0])) == 1.0


class TestTheGroundBesideABankedRoad:
    def _conformed(self, bank, **kwargs):
        path = RoadPath(_straight(200.0, height=5.0), RoadProfile(),
                        bank=np.full(21, bank))
        return path, conform_terrain(_flat, path, formation=0.0, **kwargs)

    def _verges(self, bank):
        path, ground = self._conformed(bank)
        half = path.profile.total_width / 2.0
        z = np.array([-100.0, -100.0])
        return ground(np.array([-half, half]), z)

    def test_a_level_road_has_its_two_verges_at_one_height(self) -> None:
        left, right = self._verges(0.0)
        assert left == pytest.approx(right, abs=1e-9)

    def test_a_banked_road_lifts_the_outside_verge_and_drops_the_inside(self) -> None:
        """Leaning right-side-down, the ground at -x is the high side. At one
        in ten the two verges are a two-thirds of a metre apart, which is a
        step the ground has to take rather than one it can ignore."""
        path, _ = self._conformed(0.10)
        half = path.profile.total_width / 2.0
        left, right = self._verges(0.10)         # at -half and +half
        flat_left, flat_right = self._verges(0.0)
        assert left > flat_left and right < flat_right
        # The two are a whole road's width of lean apart, less the cosine of
        # the lean, since the verge is `half` out along a surface that leans.
        assert left - right == pytest.approx(
            2.0 * 0.10 * half / np.hypot(1.0, 0.10), rel=0.02)

    def test_the_ground_under_the_crown_is_still_the_road(self) -> None:
        _, ground = self._conformed(0.10)
        assert float(ground(np.array([0.0]), np.array([-100.0]))[0]) == \
            pytest.approx(5.0, abs=1e-6)

    def test_the_earthwork_climbs_out_of_the_high_verge_without_a_step(self) -> None:
        _, ground = self._conformed(0.10)
        x = np.linspace(-60.0, 60.0, 800)
        heights = ground(x, np.full_like(x, -100.0))
        assert np.abs(np.diff(heights)).max() < 0.4

    def test_it_is_the_road_s_own_surface_at_the_carriageway_edge(self) -> None:
        path, ground = self._conformed(0.10)
        half = path.profile.carriageway_width / 2.0
        lean = 0.10
        upright = 1.0 / np.hypot(1.0, lean)
        # The pavement rotates about the crown, so the edge is `half` out along
        # a surface that leans: less than `half` out in plan.
        across = half * upright
        found = ground(np.array([across]), np.array([-100.0]))
        assert float(found[0]) == pytest.approx(5.0 - half * lean * upright,
                                                abs=1e-6)


class TestTheLeanTravelsWithTheBakedWorld:
    def _layer(self, bank):
        from OpenGLContext_editor.world.road import RoadLayer
        return RoadLayer(RoadPath(_straight(400.0, count=41, height=3.0),
                                  bank=np.full(41, bank)))

    def test_the_road_writes_its_lean_beside_its_centreline(self) -> None:
        road = self._layer(0.2).metadata()['roads'][0]
        assert len(road['bank']) == len(road['centreline'])
        assert road['bank'][10] == pytest.approx(0.2)

    def test_a_road_that_does_not_lean_says_so_rather_than_a_list_of_zeros(self) -> None:
        assert self._layer(0.0).metadata()['roads'][0]['bank'] == []

    def test_the_surface_it_writes_leans(self) -> None:
        from OpenGLContext_editor.bake.bounds import BoundingBox
        region = BoundingBox((-50, -50, -300), (50, 50, -100))
        level = self._layer(0.0).content(region, error=0.0)[0]
        leaning = self._layer(0.25).content(region, error=0.0)[0]
        assert not np.allclose(level.mesh.positions, leaning.mesh.positions)
        # +x is the road's right hand, and a positive lean puts it down.
        left = leaning.mesh.positions[leaning.mesh.positions[:, 0] < -1.0][:, 1]
        right = leaning.mesh.positions[leaning.mesh.positions[:, 0] > 1.0][:, 1]
        assert left.mean() > right.mean()


class TestWhatTheCircuitIsLaidOutFor:
    def _circuit(self, **kwargs):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(structures=False, **kwargs).circuit()

    def test_its_corners_lean(self) -> None:
        assert np.abs(self._circuit().bank).max() > 0.05

    def test_every_corner_holds_the_speed_that_stretch_is_for(self) -> None:
        """Not one speed for the whole lap: the corners are drawn from a mix
        now, and a hairpin holds what a hairpin holds. What has to be true of
        each of them is that it holds the speed it was *laid out* for."""
        from OpenGLContext.scenegraph.road import plan_curvature

        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(structures=False)
        path = world.circuit()
        curvature = np.abs(plan_curvature(path.points, closed=True))
        radius = np.where(curvature > 1e-9, 1.0 / np.maximum(curvature, 1e-12),
                          np.inf)
        held = np.array([corner_speed(float(r), bank=float(b))
                         for r, b in zip(radius, path.bank, strict=True)])
        assert np.all(held >= world.circuit_character().design_speed - 1e-6)

    def test_a_circuit_of_one_corner_holds_the_speed_it_was_laid_out_for(self) -> None:
        from OpenGLContext.scenegraph.road import plan_curvature

        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_DESIGN_SPEED,
            ProceduralWorld,
        )
        path = ProceduralWorld(structures=False, variety=0.0).circuit()
        curvature = np.abs(plan_curvature(path.points, closed=True))
        radius = np.where(curvature > 1e-9, 1.0 / np.maximum(curvature, 1e-12),
                          np.inf)
        held = np.array([corner_speed(float(r), bank=float(b))
                         for r, b in zip(radius, path.bank, strict=True)])
        assert held.min() >= CIRCUIT_DESIGN_SPEED

    def test_banking_lets_it_corner_tighter_than_a_flat_road_would(self) -> None:
        """About a fifth tighter, which is what road banking is worth. Half
        again would be an oval, and this is a road."""
        from OpenGLContext.scenegraph.road import cornering_radius

        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_DESIGN_SPEED,
            CIRCUIT_MAXIMUM_BANK,
        )
        flat = cornering_radius(CIRCUIT_DESIGN_SPEED)
        banked = cornering_radius(CIRCUIT_DESIGN_SPEED,
                                  bank=CIRCUIT_MAXIMUM_BANK)
        assert 0.7 * flat < banked < 0.9 * flat


class TestTheStartLineOnABankedRoad:
    def _placement(self, bank):
        from OpenGLContext_editor.world.gantry import start_finish
        return start_finish(RoadPath(_straight(200.0, height=5.0),
                                     RoadProfile(),
                                     bank=np.full(21, bank)), station=100.0)

    def test_a_level_road_leaves_it_level(self) -> None:
        assert self._placement(0.0).bank == pytest.approx(0.0)

    def test_it_takes_the_road_s_lean_where_the_line_is_drawn(self) -> None:
        assert self._placement(0.08).bank == pytest.approx(0.08)

    def test_the_camber_it_is_painted_on_is_what_the_lean_left(self) -> None:
        """Past the crossfall there is no crown under the paint to follow."""
        assert self._placement(0.08).crossfall == pytest.approx(0.0)
        assert self._placement(0.0).crossfall == pytest.approx(
            RoadProfile().crossfall)

    def test_the_paint_lies_on_the_carriageway_rather_than_across_it(self) -> None:
        from OpenGLContext_editor.bake.gantry import GantryLayer
        one = self._placement(0.10)
        painted = GantryLayer(placement=one)._mesh.positions
        # The line spans the carriageway, so its two ends are a carriageway's
        # width apart in plan and the lean's worth apart in height.
        line = np.asarray(painted, dtype='d')
        low = line[line[:, 1] < 5.0 - 0.3]
        assert len(low), "nothing on the low side of a road leaning one in ten"

    def test_the_legs_stay_upright(self) -> None:
        """A gantry is steel standing on two feet; the road leans, not it."""
        from OpenGLContext.scenegraph.gantry import GantryProfile

        from OpenGLContext_editor.bake.gantry import GantryLayer
        one = self._placement(0.10)
        top = np.asarray(GantryLayer(placement=one)._mesh.positions,
                         dtype='d')[:, 1].max()
        assert top == pytest.approx(5.0 + GantryProfile().height, abs=0.2)
