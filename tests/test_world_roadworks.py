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
from OpenGLContext.scenegraph.roadworks import BoreCut, TunnelProfile

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

    def test_the_hill_over_a_bore_is_left_where_it_is(self) -> None:
        """A road running *inside* the ground leaves it as it found it.

        Saying a hill is hollow is ``holes``' job
        (:meth:`OpenGLContext.scenegraph.terrain.HeightField.mesh`), so the
        earthwork has nothing to fake: it digs the portal and leaves the hill.
        """
        path, conformed = self._under(Op.TUNNEL)
        at = path.points[20]
        assert float(conformed(np.array([at[0]]), np.array([at[2]]))[0]) \
            > float(at[1]) + 1.0

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

    def test_and_no_fewer_than_a_swept_structure_can_hold(self) -> None:
        """A bore and a deck are *swept* along the tile's own resampling of the
        line, and a tube swept along points a hundred metres apart is not a
        coarse tunnel but a shape nothing in the world has. The error ladder of
        a tiled world starts at ninety-odd metres, so the spacing is held."""
        from OpenGLContext_editor.world.road import COARSEST_SPACING
        layer = self._layer(Op.BRIDGE, 60.0)
        assert layer.spacing_for(4000.0) == pytest.approx(COARSEST_SPACING)
        assert layer.spacing_for(94.0) == pytest.approx(COARSEST_SPACING)
        assert layer.spacing_for(1.0) < COARSEST_SPACING

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

    def test_the_road_records_how_its_bores_were_cut(self, world) -> None:
        """So a game cuts its collider with the mouths the tiles were cut
        with, rather than with figures of its own."""
        record = world.circuit_layer().metadata()['roads'][0]['bores']
        cut = BoreCut.from_json(record)
        assert cut == world.bore_cut()
        assert cut.tunnel.portal_border == world.tunnel_profile().portal_border
        assert cut.approach > 0.0

    def test_the_tiles_are_cut_with_the_recorded_mouths(self, world) -> None:
        cut = BoreCut.from_json(
            world.circuit_layer().metadata()['roads'][0]['bores'])
        again = cut.openings(world.circuit().tunnel_runs(), world.height_fn(),
                             profile=world.circuit().profile)
        holes = world.bore_openings()
        portal = world.circuit().portals().points
        x = portal[:, 0][:, None] + np.linspace(-20.0, 20.0, 21)[None, :]
        z = portal[:, 2][:, None] + np.linspace(-20.0, 20.0, 21)[None, :]
        assert np.array_equal(holes(x, z), again(x, z))

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

    def test_and_the_hill_the_bore_runs_through_is_still_there(self) -> None:
        """The approach is cut; what the road passes under is not."""
        path, conformed, _hill = self._approach()
        at = path.points[30]
        over = float(conformed(np.array([at[0]]), np.array([at[2]]))[0])
        assert over > float(at[1]) + 1.0

    def test_a_road_all_on_the_ground_is_unchanged(self) -> None:
        path = RoadPath(_line(height=6.0))
        conformed = conform_terrain(_ground(0.0), path)
        assert float(conformed(np.array([200.0]), np.array([0.0]))[0]) \
            == pytest.approx(6.0, abs=0.3)


class TestTheGroundAtAPortal:
    """A bore opens into a cut face, not into a knife edge of hillside.

    The cutting the road runs in stops where the bore begins and the hill takes
    over, so between one sample and the next the ground steps from the
    carriageway to the hillside -- ten or twenty metres, across a cell two
    metres wide. What stands there is a wall one sample thick with the arch cut
    out of it, and slivers of it either side of the arch read as spikes over the
    portal. The earthwork cuts the ground back around the mouth instead: down to
    the crown of the portal, and rising from there at the batter the rest of the
    cutting uses.
    """

    #: Which points of the line the bore runs between.
    FIRST, LAST = 20, 60

    def _approach(self, slope=0.35, start=30.0):
        """A road running level into a hillside, with a bore through it."""
        line = _line(height=0.0, count=81)
        path = RoadPath(line, ops=_spanning(Op.TUNNEL, self.FIRST, self.LAST,
                                            count=81))
        portal = float(line[self.FIRST][0])

        def hill(x, z):
            return np.clip((np.asarray(x, 'd') - portal + start) * slope,
                           0.0, 300.0)
        return path, conform_terrain(hill, path), hill, portal

    def crown(self):
        """How much ground the funnel leaves over the road at the mouth."""
        from OpenGLContext.scenegraph.roadworks import TunnelProfile
        tunnel = TunnelProfile()
        return tunnel.clearance + tunnel.portal_border

    def test_the_hillside_at_the_mouth_is_cut_back_to_the_portal(self) -> None:
        _path, conformed, hill, portal = self._approach(slope=1.5)
        at = portal + 4.0
        assert float(hill(at, 0.0)) > self.crown() + 10.0, 'a hill to cut'
        assert float(conformed(np.array([at]), np.array([0.0]))[0]) \
            == pytest.approx(self.crown(), abs=0.01)

    def test_what_it_leaves_stands_behind_the_face_rather_than_in_it(self) -> None:
        """A game cuts the drawn ground back to just inside the portal's face,
        so ground left at the height of the face is behind it and ground left
        above the face stands in front of it. The funnel leaves it level with
        the top of the face and no higher."""
        from OpenGLContext.scenegraph.roadworks import TunnelProfile, bore_opening
        path, conformed, _hill, portal = self._approach(slope=1.5)
        run = path.points[self.FIRST:self.LAST + 1]
        mouth = bore_opening(run, conformed, profile=path.profile,
                             tunnel=TunnelProfile(), inset=0.3)
        over = np.asarray(run[1:6])
        assert not mouth(over[:, 0], over[:, 2]).any()
        # Level with the top of the face where the funnel is flat, and above it
        # where the cut has started to rise.
        found = conformed(over[:, 0], over[:, 2]) - over[:, 1]
        assert float(np.min(found)) >= self.crown() - 0.01
        at_the_face = np.asarray(path.points[self.FIRST]) \
            + (np.asarray(run[1]) - np.asarray(run[0])) * 0.2
        assert float(conformed(np.array([at_the_face[0]]),
                               np.array([at_the_face[2]]))[0]) \
            == pytest.approx(self.crown(), abs=0.2)

    def test_the_cut_face_rises_at_the_batter(self) -> None:
        """No step between two samples steeper than the earthwork allows, from
        the face of the portal out to where the hillside takes over again.

        The first step is the face itself: the ground goes from the carriageway
        to the crown of the portal in one, because that is what a portal is, and
        the face is what stands in it.
        """
        _path, conformed, _hill, portal = self._approach()
        along = np.arange(portal + 2.0, portal + 80.0, 2.0)
        found = conformed(along, np.zeros_like(along))
        assert float(np.abs(np.diff(found)).max()) <= 2.0 * 0.6 + 0.01

    def test_it_meets_an_ordinary_hillside_inside_the_cut(self) -> None:
        """Which is why there is no step where the digging stops."""
        from OpenGLContext_editor.world.road import PORTAL_CUT
        _path, conformed, hill, portal = self._approach()
        at = portal + PORTAL_CUT - 4.0
        assert float(conformed(np.array([at]), np.array([0.0]))[0]) \
            == pytest.approx(float(hill(at, 0.0)), abs=0.01)

    def test_the_hill_the_bore_runs_through_is_still_a_hill(self) -> None:
        """Past the cut the mountain is the mountain, however steep it is: what
        is dug is the mouth, not the tunnel."""
        _path, conformed, hill, portal = self._approach(slope=1.5)
        at = portal + 200.0
        assert float(conformed(np.array([at]), np.array([0.0]))[0]) \
            == pytest.approx(float(hill(at, 0.0)), abs=0.01)

    def test_nothing_is_left_standing_over_the_carriageway(self) -> None:
        """What a car arrives at. The cutting stops where the bore begins, and
        a sample beside the carriageway a step before the face is nearer to the
        bore than to the road on the ground: left at the hillside's own height
        it is a wall across the road, and the driver hits it at speed.
        """
        path, conformed, _hill, portal = self._approach(slope=1.5)
        at = path.points[self.FIRST]
        ahead = path.points[self.FIRST] - path.points[self.FIRST - 1]
        ahead = ahead / np.hypot(ahead[0], ahead[2])
        side = np.array([-ahead[2], 0.0, ahead[0]])
        half = path.profile.total_width / 2.0
        for back in np.arange(0.5, 24.0, 0.5):
            across = np.linspace(-half, half, 9)
            points = at[None, :] - ahead[None, :] * back + side[None, :] * across[:, None]
            found = conformed(points[:, 0], points[:, 2]) - at[1]
            assert float(np.max(found)) < 0.5, (
                'ground %.1f m over the carriageway %.1f m before the portal'
                % (float(np.max(found)), back))

    def test_the_cutting_the_road_arrives_in_is_untouched(self) -> None:
        """In front of the portal the ground is what the road is laid on, and
        the funnel only ever takes ground away from above."""
        path, conformed, _hill, portal = self._approach()
        at = portal - 20.0
        assert float(conformed(np.array([at]), np.array([0.0]))[0]) \
            == pytest.approx(float(path.points[0][1]), abs=0.5)

    def test_it_leaves_the_country_either_side_alone(self) -> None:
        _path, conformed, hill, portal = self._approach()
        aside = 400.0
        assert float(conformed(np.array([portal]), np.array([aside]))[0]) \
            == pytest.approx(float(hill(portal, aside)), abs=0.01)

    def test_a_road_with_no_bore_has_no_funnel(self) -> None:
        line = _line(height=0.0, count=41)
        path = RoadPath(line)

        def hill(x, z):
            return np.full(np.shape(np.asarray(x, 'd')), 200.0)
        conformed = conform_terrain(hill, path)
        assert float(conformed(np.array([200.0]), np.array([600.0]))[0]) \
            == pytest.approx(200.0)


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

    Two things make that true and they are in two places: the hill over the
    bore is left standing here, and the mouth is taken out of the drawn ground
    by the ``holes`` mask the game builds from
    :func:`OpenGLContext.scenegraph.roadworks.bore_opening`. What is checked
    here is the first -- that there is a hillside for the mouth to be a hole in.
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

    def test_a_mouth_is_a_hole_in_a_hillside(self, world) -> None:
        """There is hill over the arch, and that is the point.

        A portal is an opening *in* something. What opens it is the ``holes``
        mask a game hands to the terrain and the collider alike
        (:func:`OpenGLContext.scenegraph.roadworks.bore_opening`); the editor's
        job is to leave the something for it to be an opening in.
        """
        circuit = world.circuit()
        ground = world.height_fn()
        line = np.asarray(circuit.points, dtype='d').reshape(-1, 3)
        stations = np.asarray(circuit.stations, dtype='d')
        for first, last in self._portals(circuit):
            middle = (first + last) / 2.0
            index = int(np.clip(np.searchsorted(stations, middle), 0,
                                len(line) - 1))
            here = line[index]
            over = float(np.asarray(
                ground(here[0], here[2])).ravel()[0]) - here[1]
            assert over > 0.0, (
                'the bore at %.0f m has no hill over it to be a bore through'
                % (middle,))


class TestTheHillOverABore:
    """A tunnel runs *under* the hill; the hill stays where it is.

    A height field is a surface, so what says a hill is hollow is ``holes`` --
    taken by the drawn mesh and the collider alike. The earthwork digs the
    portals and leaves the hill between them, which is what makes the bore a
    bore rather than a valley with a lid.
    """

    @pytest.fixture(scope='class')
    def world(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=4096.0)

    def bored(self, world):
        circuit = world.circuit()
        inside = np.array([op is Op.TUNNEL for op in circuit.ops], dtype=bool)
        if not inside.any():
            pytest.skip('this world tunnels through nothing')
        return circuit, inside

    def test_no_segment_inside_a_bore_reshapes_the_ground(self, world) -> None:
        circuit, inside = self.bored(world)
        both_ends = inside[:-1] & inside[1:]
        assert not circuit.reshaped_segments()[both_ends].any()

    def test_the_road_on_the_land_still_does(self, world) -> None:
        """The approach cuttings are the whole point of an earthwork."""
        circuit, _ = self.bored(world)
        assert circuit.reshaped_segments()[circuit.segment_on_ground].all()

    def test_the_mountain_is_still_a_mountain(self, world) -> None:
        """Over a bore the ground is the ground: what a tunnel runs through is
        a hill, and the road is under it.

        Measured away from the portals, since a portal *is* dug -- see
        :class:`TestTheGroundAtAPortal` for the funnel and how far it reaches.
        """
        from OpenGLContext_editor.world.road import PORTAL_CUT
        circuit, inside = self.bored(world)
        line = circuit.points
        portals = circuit.portals().points
        gap = np.hypot(line[:, None, 0] - portals[None, :, 0],
                       line[:, None, 2] - portals[None, :, 2]).min(axis=1)
        over = inside & (gap > PORTAL_CUT)
        if not over.any():
            pytest.skip('every bore here is shorter than its own portal cuts')
        natural = np.asarray(world.natural()(line[:, 0], line[:, 2]), dtype='d')
        conformed = np.asarray(world.height_fn()(line[:, 0], line[:, 2]),
                               dtype='d')
        dropped = natural[over] - conformed[over]
        assert float(np.percentile(dropped, 90)) < 5.0, (
            'the hill over a bore is still being pushed down to the road')

    def test_the_ground_over_a_bore_is_above_the_road(self, world) -> None:
        """Which is what "the road runs inside the hill" means."""
        circuit, inside = self.bored(world)
        line = circuit.points
        conformed = np.asarray(world.height_fn()(line[:, 0], line[:, 2]),
                               dtype='d')
        cover = conformed[inside] - line[inside, 1]
        assert float(np.median(cover)) > 2.0


class TestTheOpeningsABoresMouthNeeds:
    """A tiled world's ground is meshed at bake time, so the hole a portal
    needs is cut there rather than by the game
    (:func:`OpenGLContext.scenegraph.roadworks.bore_opening`)."""

    def _path(self, kind=Op.TUNNEL):
        return RoadPath(_line(height=-40.0), ops=_spanning(kind, 10, 30))

    def _ridge(self):
        """A hill the road runs under: the surface passes through each portal
        and closes right over the middle of the bore."""
        def at(x, z):
            x = np.asarray(x, dtype='d')
            over = 60.0 * np.clip(1.0 - np.abs(x - 200.0) / 100.0, 0.0, 1.0)
            return np.broadcast_to(
                -38.0 + over,
                np.broadcast(x, np.asarray(z, dtype='d')).shape)
        return at

    def test_a_road_with_no_bore_opens_nothing(self) -> None:
        assert self._path(Op.BRIDGE).bore_openings(self._ridge()) is None

    def test_a_bore_opens_at_its_mouths(self) -> None:
        holes = self._path().bore_openings(self._ridge())
        assert holes is not None
        mouth = self._path().points[10]
        assert bool(np.asarray(holes(np.array([mouth[0]]),
                                     np.array([mouth[2]])))[0])

    def test_and_nowhere_the_hill_has_closed_over_it(self) -> None:
        """The hillside over the middle of a bore is hillside."""
        holes = self._path().bore_openings(self._ridge())
        middle = self._path().points[20]
        assert not bool(np.asarray(holes(np.array([middle[0]]),
                                         np.array([middle[2]])))[0])

    def test_nor_out_on_the_open_ground(self) -> None:
        holes = self._path().bore_openings(self._ridge())
        assert not np.asarray(holes(np.array([200.0]), np.array([400.0]))).any()

    def test_the_answer_keeps_the_shape_it_was_asked_in(self) -> None:
        holes = self._path().bore_openings(self._ridge())
        assert np.asarray(holes(np.zeros((3, 4)), np.zeros((3, 4)))).shape \
            == (3, 4)

    def test_a_wider_approach_clears_more_of_the_cutting(self) -> None:
        path = self._path()
        near = path.bore_openings(self._ridge(), BoreCut(approach=1.0))
        far = path.bore_openings(self._ridge(), BoreCut(approach=30.0))
        x = np.linspace(0.0, 400.0, 401)
        z = np.zeros_like(x)
        assert int(np.asarray(far(x, z)).sum()) \
            > int(np.asarray(near(x, z)).sum())


class TestABoreAcrossTheStartOfACircuit:
    """A closed circuit's line begins and ends at one point, so a bore through
    the start line is one bore whose run the line's two ends split."""

    def _circuit(self, count=73):
        turn = np.linspace(0.0, 2.0 * np.pi, count)
        points = np.stack([500.0 * np.cos(turn), np.zeros(count),
                           500.0 * np.sin(turn)], axis=-1)
        points[-1] = points[0]
        ops = [Op.DIRT] * count
        for index in list(range(0, 6)) + list(range(count - 6, count)):
            ops[index] = Op.TUNNEL
        return RoadPath(points, ops=ops)

    def test_it_has_two_portals(self) -> None:
        portals = self._circuit().portals()
        assert len(portals.points) == 2
        path = self._circuit()
        assert {tuple(np.round(one, 3)) for one in portals.points} == {
            tuple(np.round(path.points[5], 3)),
            tuple(np.round(path.points[-6], 3))}

    def test_its_run_is_one_run(self) -> None:
        runs = self._circuit().tunnel_runs()
        assert len(runs) == 1
        assert len(runs[0]) == 11             # 6 + 6, the shared point once

    def test_an_open_road_s_ends_are_its_own(self) -> None:
        path = RoadPath(_line(), ops=_spanning(Op.TUNNEL, 0, 5))
        assert len(path.tunnel_runs()) == 1
        assert len(path.portals().points) == 2


class TestAPortalOnAHairpin:
    """A road that doubles back may pass beside its own portal on another
    stretch at another height. The ground at the portal is cut from the bore's
    own road, not from whichever stretch is nearest."""

    def _path(self):
        approach = [(-100.0 + 10.0 * i, 0.0, 0.0) for i in range(10)]
        bore = [(10.0 * i, 0.0, 0.0) for i in range(11)]
        turn = [(100.0, 5.0, 30.0)]
        above = [(100.0 - 10.0 * i, 20.0, 15.0) for i in range(21)]
        points = np.asarray(approach + bore + turn + above, dtype='d')
        ops = ([Op.DIRT] * 10 + [Op.TUNNEL] * 11 + [Op.BRIDGE]
               + [Op.BRIDGE] * 21)
        return RoadPath(points, ops=ops)

    def test_the_cut_hangs_from_the_bore_s_own_road(self) -> None:
        path = self._path()
        high = _ground(50.0)
        conformed = conform_terrain(high, path)
        # Nearer the stretch overhead (7 m) than the portal (11 m).
        at_x, at_z = np.array([-8.0]), np.array([8.0])
        assert float(path.sample(at_x, at_z, radius=100.0).height[0]) \
            == pytest.approx(20.0, abs=1.0)
        tunnel = TunnelProfile()
        crown = tunnel.clearance + tunnel.portal_border
        assert float(conformed(at_x, at_z)[0]) < 20.0
        assert float(conformed(at_x, at_z)[0]) >= crown - 1e-6
