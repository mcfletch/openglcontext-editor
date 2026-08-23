"""A road with structures on it, and what the world does about them.

The choice of what to build is :mod:`OpenGLContext_editor.world.structures` and
the geometry is :mod:`OpenGLContext.scenegraph.roadworks`. What is tested here is
the join between them: a road path that knows which of its stretches stand on
the land, ground that is left alone under a deck and over a bore, and a bake
layer that writes the deck and the bore into the tiles they cross.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.road import RoadLayer, RoadPath, conform_terrain
from OpenGLContext_editor.world.structures import Op

SPACING = 10.0


def _line(height=0.0, count=41, spacing=SPACING):
    x = np.arange(count) * spacing
    return np.stack([x, np.full(count, float(height)), np.zeros(count)], axis=-1)


def _ground(level=0.0):
    def at(x, z):
        return np.full(np.broadcast(np.asarray(x), np.asarray(z)).shape,
                       float(level), dtype='d')
    return at


def _spanning(kind, first, last, count=41):
    """A path whose middle stretch is carried by a structure."""
    ops = [Op.DIRT] * count
    for i in range(first, last + 1):
        ops[i] = kind
    return ops


class TestAPathThatKnowsItsStructures:
    def test_a_plain_path_stands_on_the_ground_all_the_way(self) -> None:
        path = RoadPath(_line())
        assert bool(path.on_ground.all())

    def test_a_path_told_about_a_bridge_does_not(self) -> None:
        path = RoadPath(_line(), ops=_spanning(Op.BRIDGE, 10, 30))
        assert not path.on_ground[20]
        assert path.on_ground[0]

    def test_a_causeway_is_carried_too(self) -> None:
        """Its fill is a structure the width of the road, not shaped land: the
        terrain either side of it is left where it was found."""
        path = RoadPath(_line(), ops=_spanning(Op.CAUSEWAY, 10, 30))
        assert not path.on_ground[20]
        assert path.on_ground[0]

    def test_the_ops_have_to_match_the_points(self) -> None:
        with pytest.raises(ValueError):
            RoadPath(_line(count=41), ops=[Op.DIRT] * 3)

    def test_a_sample_says_which_segment_answered_it(self) -> None:
        path = RoadPath(_line())
        found = path.sample(np.array([105.0]), np.array([0.0]))
        assert int(found.segment[0]) == 10

    def test_a_sample_agrees_with_the_plain_query(self) -> None:
        path = RoadPath(_line(height=12.0))
        x = np.array([15.0, 205.0, 390.0])
        z = np.array([4.0, -30.0, 0.0])
        found = path.sample(x, z)
        distance, height = path.nearest(x, z)
        assert np.allclose(found.distance, distance)
        assert np.allclose(found.height, height)

    def test_a_sample_keeps_the_shape_it_was_asked_in(self) -> None:
        path = RoadPath(_line())
        found = path.sample(np.zeros((3, 4)), np.zeros((3, 4)))
        assert found.distance.shape == (3, 4)
        assert found.segment.shape == (3, 4)

    def test_a_query_out_of_reach_is_still_answered(self) -> None:
        path = RoadPath(_line())
        found = path.sample(np.array([9000.0]), np.array([0.0]), radius=50.0)
        assert not np.isfinite(found.distance[0])


class TestGroundUnderAStructure:
    def _under(self, kind):
        path = RoadPath(_line(height=60.0 if kind is Op.BRIDGE else -60.0),
                        ops=_spanning(kind, 10, 30))
        return path, conform_terrain(_ground(0.0), path)

    def test_the_land_under_a_deck_is_left_as_it_was(self) -> None:
        _path, conformed = self._under(Op.BRIDGE)
        assert float(conformed(np.array([200.0]), np.array([0.0]))[0]) \
            == pytest.approx(0.0)

    def test_the_hill_over_a_bore_comes_out_from_under_it(self) -> None:
        """A ground mesh is a surface, so the only way to take a tunnel out of
        it is to cut down to the road for the length of the bore. Left in, the
        hillside stands inside the tube and a driver looking into the portal
        sees the hill rather than the lining."""
        path, conformed = self._under(Op.TUNNEL)
        at = path.points[20]
        assert float(conformed(np.array([at[0]]), np.array([at[2]]))[0]) \
            <= float(at[1]) + 1e-6

    def test_and_the_land_out_past_the_cutting_is_left_as_it_was(self) -> None:
        _path, conformed = self._under(Op.TUNNEL)
        assert float(conformed(np.array([200.0]), np.array([400.0]))[0]) \
            == pytest.approx(0.0)

    def test_the_ground_beside_a_deck_is_left_as_it_was(self) -> None:
        _path, conformed = self._under(Op.BRIDGE)
        assert float(conformed(np.array([200.0]), np.array([40.0]))[0]) \
            == pytest.approx(0.0)

    def test_where_the_road_is_on_dirt_the_ground_still_meets_it(self) -> None:
        path = RoadPath(_line(height=6.0), ops=_spanning(Op.BRIDGE, 10, 30))
        conformed = conform_terrain(_ground(0.0), path)
        on_road = float(conformed(np.array([20.0]), np.array([0.0]))[0])
        assert on_road == pytest.approx(6.0, abs=0.3)

    def test_a_road_with_no_structures_conforms_everywhere(self) -> None:
        path = RoadPath(_line(height=6.0))
        conformed = conform_terrain(_ground(0.0), path)
        assert float(conformed(np.array([200.0]), np.array([0.0]))[0]) \
            == pytest.approx(6.0, abs=0.3)

    def test_a_whole_grid_of_ground_is_answered_at_once(self) -> None:
        path = RoadPath(_line(height=60.0), ops=_spanning(Op.BRIDGE, 10, 30))
        conformed = conform_terrain(_ground(0.0), path)
        axis = np.linspace(0.0, 400.0, 41)
        gx, gz = np.meshgrid(axis, np.linspace(-100.0, 100.0, 21), indexing='ij')
        found = conformed(gx, gz)
        assert found.shape == gx.shape
        # Under the span, nothing moved.
        middle = found[(gx > 120.0) & (gx < 280.0)]
        assert float(np.abs(middle).max()) == pytest.approx(0.0, abs=1e-9)


class TestTheLayerWritesTheStructures:
    def _layer(self, kind, height):
        line = _line(height=height, count=41)
        path = RoadPath(line, ops=_spanning(kind, 10, 30))
        return RoadLayer(path, ground=_ground(0.0))

    def _tile(self):
        return BoundingBox((-50.0, -200.0, -100.0), (450.0, 200.0, 100.0))

    def test_a_bridge_is_written_into_the_tile_it_crosses(self) -> None:
        found = self._layer(Op.BRIDGE, 60.0).content(self._tile(), 1.0)
        assert any('bridge' in node.name for node in found)

    def test_a_tunnel_is_written_into_the_tile_it_crosses(self) -> None:
        found = self._layer(Op.TUNNEL, -60.0).content(self._tile(), 1.0)
        assert any('tunnel' in node.name for node in found)

    def test_the_carriageway_is_written_too(self) -> None:
        found = self._layer(Op.BRIDGE, 60.0).content(self._tile(), 1.0)
        assert any(node.name == 'road' for node in found)

    def test_the_deck_sits_under_the_carriageway(self) -> None:
        found = self._layer(Op.BRIDGE, 60.0).content(self._tile(), 1.0)
        deck = next(n for n in found if n.name.endswith('deck'))
        road = next(n for n in found if n.name == 'road')
        assert float(deck.mesh.positions[:, 1].max()) \
            <= float(road.mesh.positions[:, 1].max()) + 0.01

    def test_the_bore_stands_over_the_carriageway(self) -> None:
        found = self._layer(Op.TUNNEL, -60.0).content(self._tile(), 1.0)
        bore = next(n for n in found if n.name.endswith('bore'))
        road = next(n for n in found if n.name == 'road')
        assert float(bore.mesh.positions[:, 1].max()) \
            > float(road.mesh.positions[:, 1].max()) + 3.0

    def test_a_road_with_no_structures_writes_only_the_carriageway(self) -> None:
        layer = RoadLayer(RoadPath(_line(height=4.0)), ground=_ground(0.0))
        found = layer.content(self._tile(), 1.0)
        assert {node.name for node in found} == {'road'}

    def test_a_tile_the_structure_misses_gets_none_of_it(self) -> None:
        layer = self._layer(Op.BRIDGE, 60.0)
        near_end = BoundingBox((350.0, -200.0, -100.0), (450.0, 200.0, 100.0))
        assert not any('bridge' in node.name
                       for node in layer.content(near_end, 1.0))

    def test_the_carriageway_over_a_deck_has_no_falling_verge(self) -> None:
        """It is an edge beam with a parapet on it, not a grass bank."""
        found = self._layer(Op.BRIDGE, 60.0).content(self._tile(), 1.0)
        road = next(n for n in found if n.name == 'road').mesh.positions
        over = road[(road[:, 0] > 150.0) & (road[:, 0] < 250.0)]
        assert len(over)
        drop = float(over[:, 1].max() - over[:, 1].min())
        assert drop < RoadProfile().verge_drop / 2.0

    def test_the_carriageway_away_from_it_keeps_its_verge(self) -> None:
        found = self._layer(Op.BRIDGE, 60.0).content(self._tile(), 1.0)
        road = next(n for n in found if n.name == 'road').mesh.positions
        away = road[road[:, 0] < 40.0]
        drop = float(away[:, 1].max() - away[:, 1].min())
        assert drop == pytest.approx(RoadProfile().verge_drop, abs=0.3)

    def test_a_distant_tile_carries_the_structure_at_fewer_vertices(self) -> None:
        layer = self._layer(Op.BRIDGE, 60.0)
        fine = layer.content(self._tile(), 1.0)
        coarse = layer.content(self._tile(), 40.0)
        assert sum(len(n.mesh.positions) for n in coarse) \
            < sum(len(n.mesh.positions) for n in fine)

    def test_without_ground_the_deck_is_still_written(self) -> None:
        line = _line(height=60.0, count=41)
        layer = RoadLayer(RoadPath(line, ops=_spanning(Op.BRIDGE, 10, 30)))
        assert any('bridge' in node.name
                   for node in layer.content(self._tile(), 1.0))


class TestWhatTheGameIsTold:
    def test_the_structures_travel_with_the_world(self) -> None:
        layer = RoadLayer(RoadPath(_line(height=60.0),
                                   ops=_spanning(Op.BRIDGE, 10, 30)))
        road = layer.metadata()['roads'][0]
        assert road['structures']

    def test_each_one_says_what_it_is_and_where(self) -> None:
        layer = RoadLayer(RoadPath(_line(height=60.0),
                                   ops=_spanning(Op.BRIDGE, 10, 30)))
        found = layer.metadata()['roads'][0]['structures'][0]
        assert found['kind'] == 'bridge'
        assert found['from'] == pytest.approx(100.0, abs=SPACING)
        assert found['to'] == pytest.approx(300.0, abs=SPACING)

    def test_a_road_all_on_dirt_reports_no_structures(self) -> None:
        layer = RoadLayer(RoadPath(_line()))
        assert layer.metadata()['roads'][0]['structures'] == []

    def test_it_is_plain_json(self) -> None:
        import json
        layer = RoadLayer(RoadPath(_line(height=60.0),
                                   ops=_spanning(Op.TUNNEL, 10, 30)))
        assert 'tunnel' in json.dumps(layer.metadata())


class TestTheShippedWorld:
    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=4096.0)

    def test_its_circuit_has_structures_on_it(self, world) -> None:
        found = world.circuit().structure_runs()
        assert any(kind is Op.BRIDGE for kind, _from, _to in found)
        assert any(kind is Op.TUNNEL for kind, _from, _to in found)

    def test_the_road_path_knows_about_them(self, world) -> None:
        assert not bool(world.circuit().on_ground.all())

    def test_turning_them_off_leaves_the_road_on_dirt(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        plain = ProceduralWorld(extent=4096.0, structures=False)
        assert bool(plain.circuit().on_ground.all())

    def test_the_ground_is_no_longer_moved_by_the_height_of_a_mountain(
            self, world) -> None:
        """What the structures are for: the earthworks left over are the size
        of earthworks.

        Measured where the road is on the land and under a deck. A bore is the
        one place the ground *is* moved by the height of a mountain, because a
        surface cannot hold a tunnel any other way -- see
        :meth:`~OpenGLContext_editor.world.road.RoadPath.reshaped_segments`.
        """
        terrain_height = world.natural()
        circuit = world.circuit()
        conformed = world.height_fn()
        line = circuit.points
        natural = np.asarray(terrain_height(line[:, 0], line[:, 2]), dtype='d')
        # Just outside the road corridor, where the earthwork does its work.
        half = circuit.profile.total_width / 2.0 + 4.0
        right = np.stack([line[:, 0], line[:, 2]], axis=-1)
        offset = np.zeros_like(right)
        step = np.diff(line[:, [0, 2]], axis=0, append=line[:1, [0, 2]])
        norm = np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
        offset[:, 0] = -step[:, 1] / norm[:, 0] * half
        offset[:, 1] = step[:, 0] / norm[:, 0] * half
        beside = right + offset
        moved = np.abs(np.asarray(conformed(beside[:, 0], beside[:, 1]))
                       - np.asarray(terrain_height(beside[:, 0], beside[:, 1])))
        bored = np.array([op is Op.TUNNEL for op in circuit.ops], dtype=bool)
        assert float(np.percentile(moved[~bored], 99)) < 40.0
        # And there was something worth building a structure for.
        assert float(np.abs(line[:, 1] - natural).max()) > 40.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestWhereAStructureMeetsTheGround:
    """The last stretch of road before a portal is on the ground, and the ground
    has to come down to meet it. Reshaping only where *both* ends of a segment
    are laid leaves the approach untouched -- a lip of hillside across the road
    at the very place a car arrives at speed."""

    def _approach(self):
        """Road running level into a hill, with a bore through it."""
        line = _line(height=0.0, count=41)
        path = RoadPath(line, ops=_spanning(Op.TUNNEL, 20, 40))

        def hill(x, z):
            return np.clip((np.asarray(x, 'd') - 150.0) * 0.4, 0.0, 60.0)
        return path, conform_terrain(hill, path), hill

    def test_the_ground_meets_the_road_right_up_to_the_portal(self) -> None:
        path, conformed, _hill = self._approach()
        # Beside the carriageway, a step before the bore begins.
        at = path.points[19]
        beside = float(conformed(np.array([at[0]]), np.array([at[2] + 4.0]))[0])
        assert beside < at[1] + 0.5

    def test_the_hill_out_past_the_cutting_is_still_there(self) -> None:
        path, conformed, hill = self._approach()
        at = path.points[30]
        aside = at[2] + 400.0
        over = float(conformed(np.array([at[0]]), np.array([aside]))[0])
        assert over == pytest.approx(float(hill(at[0], aside)), abs=0.01)

    def test_and_the_bore_itself_is_cut_down_to_the_road(self) -> None:
        path, conformed, _hill = self._approach()
        at = path.points[30]
        over = float(conformed(np.array([at[0]]), np.array([at[2]]))[0])
        assert over <= float(at[1]) + 1e-6

    def test_a_road_all_on_the_ground_is_unchanged(self) -> None:
        path = RoadPath(_line(height=6.0))
        conformed = conform_terrain(_ground(0.0), path)
        assert float(conformed(np.array([200.0]), np.array([0.0]))[0]) \
            == pytest.approx(6.0, abs=0.3)


class TestTheRoadsOwnSectionTravelsWithIt:
    """A game that builds its own collider for the carriageway -- because tile
    geometry changes resolution under a car and the surface must not -- needs
    the cut, not just how wide it is."""

    def _profile(self):
        layer = RoadLayer(RoadPath(_line()))
        return layer.metadata()['roads'][0]['profile']

    def test_the_cut_is_written_out(self) -> None:
        assert self._profile()

    def test_it_says_what_the_carriageway_is_made_of(self) -> None:
        found = self._profile()
        assert found['laneWidth'] == pytest.approx(RoadProfile().lane_width)
        assert found['lanes'] == RoadProfile().lanes

    def test_it_says_what_is_beside_it(self) -> None:
        found = self._profile()
        default = RoadProfile()
        assert found['shoulderWidth'] == pytest.approx(default.shoulder_width)
        assert found['vergeWidth'] == pytest.approx(default.verge_width)
        assert found['vergeDrop'] == pytest.approx(default.verge_drop)

    def test_it_rebuilds_the_profile_it_came_from(self) -> None:
        line = _line()
        mine = RoadProfile(lane_width=3.2, lanes=2, shoulder_width=0.6,
                           verge_width=0.9, verge_drop=0.4, crossfall=0.03,
                           texture_length=18.0)
        found = RoadLayer(RoadPath(line, profile=mine)).metadata()['roads'][0]
        import numpy as np
        rebuilt = RoadProfile(
            lane_width=found['profile']['laneWidth'],
            lanes=found['profile']['lanes'],
            shoulder_width=found['profile']['shoulderWidth'],
            shoulder_drop=found['profile']['shoulderDrop'],
            verge_width=found['profile']['vergeWidth'],
            verge_drop=found['profile']['vergeDrop'],
            crossfall=found['profile']['crossfall'],
            texture_length=found['profile']['textureLength'])
        assert np.allclose(rebuilt.section(), mine.section())

    def test_it_is_plain_json(self) -> None:
        import json
        assert json.loads(json.dumps(self._profile()))


class TestAPortalIsOpen:
    """A driver arriving at a bore has to be able to see into it.

    The ground a tunnel passes through is cut down to the road so the lining
    stands in a cutting -- but a ground mesh draws straight lines between its
    samples, so it is the sample *at the mouth* that decides what the opening
    looks like. Left at the hillside's own height it is a wall across the road
    with the arch in the air behind it, and the only reason a car gets through
    is that the collider has the bore taken out of it.
    """

    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=2048.0, seed=11)

    @staticmethod
    def _portals(circuit):
        return [(first, last) for kind, first, last in circuit.structure_runs()
                if kind is Op.TUNNEL]

    def test_the_world_has_bores_to_look_into(self, world) -> None:
        assert self._portals(world.circuit())

    def test_nothing_stands_over_the_arch_at_the_mouth(self, world) -> None:
        from OpenGLContext.scenegraph.roadworks import TunnelProfile
        circuit = world.circuit()
        ground = world.height_fn()
        line = np.asarray(circuit.points, dtype='d').reshape(-1, 3)
        stations = np.asarray(circuit.stations, dtype='d')
        clearance = TunnelProfile().clearance
        for first, last in self._portals(circuit):
            for at in (first, last):
                index = int(np.clip(np.searchsorted(stations, at), 0,
                                    len(line) - 1))
                here = line[index]
                over = float(np.asarray(
                    ground(here[0], here[2])).ravel()[0]) - here[1]
                assert over <= clearance, (
                    'the mouth at %.0f m has %.1f m of hill over the arch'
                    % (at, over - clearance))
