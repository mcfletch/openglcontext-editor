"""A causeway is a structure the road is carried on, not a shape of the land.

Built as earthworks, a road three metres over a lake margin drags the terrain up
with it and the batter reaches a hundred metres either side -- a ridge across
the water, not a crossing over it. Built as a structure, the land is left where
it was found and the road rides a walled embankment its own width.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.road import CARRIED, RoadLayer, RoadPath
from OpenGLContext_editor.world.structures import Op

PROFILE = RoadProfile(lane_width=3.6, lanes=2, shoulder_width=0.7,
                      shoulder_drop=0.05, verge_width=1.0, verge_drop=0.35)


def _crossing(length=200.0, points=41):
    z = np.linspace(0.0, length, points)
    line = np.stack([np.zeros(points), np.full(points, 20.0), z], axis=-1)
    ops = np.full(points, Op.DIRT, dtype=object)
    ops[10:31] = Op.CAUSEWAY
    return RoadPath(line, profile=PROFILE, ops=ops)


def _ground(x, z):
    """Twenty metres up, except across the middle, where it drops away."""
    z = np.asarray(z, 'd')
    return np.where((z > 45.0) & (z < 155.0), 12.0, 20.0)


class TestItIsCarried:
    def test_a_causeway_is_a_structure(self) -> None:
        assert Op.CAUSEWAY in CARRIED

    def test_the_land_under_it_is_left_alone(self) -> None:
        path = _crossing()
        assert not path.on_ground[20]

    def test_the_road_either_side_is_still_laid(self) -> None:
        path = _crossing()
        assert path.on_ground[0] and path.on_ground[-1]


class TestWhatABakeWrites:
    def _nodes(self):
        layer = RoadLayer(_crossing(), ground=_ground)
        return layer.content(BoundingBox((-200.0, -50.0, -50.0),
                                         (200.0, 100.0, 250.0)), error=1.0)

    def test_it_writes_a_body_and_a_wall(self) -> None:
        named = {node.name for node in self._nodes()}
        assert 'causeway-body' in named
        assert 'causeway-wall' in named

    def test_the_wall_stands_above_the_road(self) -> None:
        for node in self._nodes():
            if node.name == 'causeway-wall':
                assert float(node.mesh.positions[:, 1].max()) > 20.2
                return
        raise AssertionError("no wall was written")

    def test_the_fill_reaches_the_ground_below(self) -> None:
        for node in self._nodes():
            if node.name == 'causeway-body':
                assert float(node.mesh.positions[:, 1].min()) < 13.0
                return
        raise AssertionError("no body was written")

    def test_it_is_only_as_wide_as_the_road_it_carries(self) -> None:
        for node in self._nodes():
            if node.name == 'causeway-body':
                assert float(node.mesh.positions[:, 0].max()) < 6.5
                return
        raise AssertionError("no body was written")

    def test_the_course_says_where_it_is(self) -> None:
        found = RoadLayer(_crossing()).metadata()['roads'][0]['structures']
        assert [one['kind'] for one in found] == ['causeway']


class TestTheShippedWorld:
    def test_its_crossings_are_built_as_structures(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        circuit = ProceduralWorld().circuit()
        kinds = {kind for kind, _, _ in circuit.structure_runs()}
        assert Op.CAUSEWAY in kinds


class TestWhatTheWallIsMadeOf:
    """A causeway's wall is the same stuff as the fill it stands on.

    What stands on the edge of a causeway is a low solid wall, not a railing,
    so the barrier material is the wrong thing on it. That material is dark on
    purpose -- a viaduct's parapet is the thing closest to the camera for a
    whole span, and structural concrete there comes out white -- and a vertical
    face at a tenth albedo is lit by nothing but sky whichever way the sun is.
    Seen from the driver's seat, on both sides of the crossing at once, that is
    a black line lying along the horizon for as long as the causeway lasts.
    """

    def _named(self, kind='causeway', **named):
        layer = RoadLayer(_crossing(), ground=_ground, **named)
        for node in layer.content(BoundingBox((-200.0, -50.0, -50.0),
                                              (200.0, 100.0, 250.0)), 1.0):
            if node.name == '%s-wall' % (kind,):
                return node.mesh.material
        raise AssertionError('no %s wall was written' % (kind,))

    def test_the_wall_is_the_concrete_the_fill_is(self) -> None:
        from OpenGLContext.scenegraph.roadworks import CONCRETE_ALBEDO
        assert tuple(float(one) for one in self._named().baseColor[:3]) \
            == pytest.approx(CONCRETE_ALBEDO, abs=1e-6)

    def test_a_caller_may_choose_its_own(self) -> None:
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        wanted = PBRMaterial(baseColor=(0.4, 0.1, 0.1))
        assert self._named(structure_material=wanted) is wanted

    def test_and_the_barrier_material_is_left_for_a_railing(self) -> None:
        """Which is a deck's parapet, and not a causeway's wall."""
        from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
        wanted = PBRMaterial(baseColor=(0.4, 0.1, 0.1))
        assert self._named(barrier=wanted) is not wanted


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
