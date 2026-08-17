"""Putting a road into a world: the alignment, the earthworks, and the tiles.

The engine's tests cover the road's *shape*. These cover what a world does
about it -- where the line ends up after it is settled onto the ground, how the
terrain is reshaped to meet it, and what each tile gets written.
"""

from itertools import pairwise

import numpy as np
import pytest
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.road import (
    RoadLayer,
    RoadPath,
    conform_terrain,
    follow_terrain,
)
from OpenGLContext_editor.world.road import curvature_limit as _curvature_limit


def _bumpy(x, z):
    """Ground with a broad slope and a metre-scale ripple on top of it."""
    x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
    return 0.05 * x + 3.0 * np.sin(z * 0.5) + 8.0 * np.sin(x * 0.01)


def _flat(x, z):
    return np.zeros_like(np.asarray(x, 'd'))


def _straight(length=200.0, count=21, height=0.0):
    z = np.linspace(0.0, -length, count)
    return np.stack([np.zeros(count), np.full(count, height), z], axis=-1)


class TestTheCentreline:
    def test_it_measures_its_own_length(self) -> None:
        assert RoadPath(_straight(200.0)).length == pytest.approx(200.0)

    def test_its_bounds_include_the_road_s_width(self) -> None:
        path = RoadPath(_straight(), RoadProfile())
        assert path.bounds().minimum[0] < -path.profile.total_width / 2.0 + 1e-9

    def test_a_line_of_one_point_is_refused(self) -> None:
        with pytest.raises(ValueError, match='two points'):
            RoadPath(np.array([(0, 0, 0.0)]))

    def test_it_resamples_for_a_coarser_tile(self) -> None:
        path = RoadPath(_straight(200.0))
        assert len(path.resampled(50.0)) < len(path.resampled(5.0))


class TestDistanceFromTheRoad:
    def _path(self):
        return RoadPath(_straight(200.0, height=7.0))

    def test_a_point_on_the_road_is_at_no_distance(self) -> None:
        distance, _ = self._path().nearest(0.0, -100.0)
        assert float(distance) == pytest.approx(0.0, abs=1e-9)

    def test_distance_is_measured_across_the_ground(self) -> None:
        """The road is seven metres up; a point beside it is 20 m away, not 21."""
        distance, _ = self._path().nearest(20.0, -100.0)
        assert float(distance) == pytest.approx(20.0)

    def test_the_height_reported_is_the_road_s(self) -> None:
        _, height = self._path().nearest(20.0, -100.0)
        assert float(height) == pytest.approx(7.0)

    def test_past_the_end_it_measures_to_the_end(self) -> None:
        distance, _ = self._path().nearest(0.0, -230.0)
        assert float(distance) == pytest.approx(30.0)

    def test_it_answers_a_whole_grid_at_once(self) -> None:
        x, z = np.meshgrid(np.linspace(-50, 50, 9), np.linspace(-200, 0, 9))
        distance, height = self._path().nearest(x, z)
        assert distance.shape == x.shape and height.shape == x.shape

    def test_a_radius_leaves_the_far_field_alone(self) -> None:
        distance, _ = self._path().nearest(np.array([500.0]), np.array([-100.0]),
                                           radius=20.0)
        assert not np.isfinite(distance[0])

    def test_a_climbing_road_reports_the_height_where_you_are(self) -> None:
        line = np.array([(0, 0, 0), (0, 100.0, -100.0)])
        distance, height = RoadPath(line).nearest(5.0, -50.0)
        assert float(height) == pytest.approx(50.0, abs=1e-6)
        assert float(distance) == pytest.approx(5.0, abs=1e-6)

    def test_the_road_crossing_a_region_is_detected(self) -> None:
        path = self._path()
        assert path.crosses(BoundingBox((-10, -100, -150), (10, 100, -50)))
        assert not path.crosses(BoundingBox((400, -100, -150), (500, 100, -50)))


class TestTheAlignment:
    def test_it_settles_onto_the_ground(self) -> None:
        line = follow_terrain([(0, 0), (0, -200)], _bumpy, spacing=5.0, smoothing=0.0)
        assert np.allclose(line[:, 1], _bumpy(line[:, 0], line[:, 2]), atol=1e-9)

    def test_smoothing_takes_the_ripple_out(self) -> None:
        rough = follow_terrain([(0, 0), (0, -200)], _bumpy, spacing=5.0, smoothing=0.0)
        smooth = follow_terrain([(0, 0), (0, -200)], _bumpy, spacing=5.0, smoothing=60.0)
        assert np.std(np.diff(smooth[:, 1])) < np.std(np.diff(rough[:, 1])) * 0.5

    def test_smoothing_keeps_the_ends_where_they_were(self) -> None:
        line = follow_terrain([(0, 0), (0, -200)], _bumpy, spacing=5.0, smoothing=60.0)
        assert line[0, 1] == pytest.approx(_bumpy(0.0, 0.0), abs=3.0)

    def test_a_plan_may_be_given_as_xz(self) -> None:
        line = follow_terrain(np.array([(0.0, 0.0), (0.0, -50.0)]), _flat, spacing=10.0)
        assert line.shape[1] == 3

    def test_clearance_lifts_the_whole_alignment(self) -> None:
        line = follow_terrain([(0, 0), (0, -50)], _flat, spacing=10.0, clearance=2.5)
        assert np.allclose(line[:, 1], 2.5)

    def test_a_grade_limit_is_honoured(self) -> None:
        def cliff(x, z):
            return np.where(np.asarray(z, 'd') < -100.0, 100.0, 0.0)

        line = follow_terrain([(0, 0), (0, -200)], cliff, spacing=5.0, smoothing=0.0,
                              maximum_grade=0.08)
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        grades = np.abs(np.diff(line[:, 1])) / steps
        assert grades.max() <= 0.08 + 1e-9

    def test_a_grade_limit_smooths_both_approaches(self) -> None:
        """A peak too steep to climb is also too steep to come down."""
        def hill(x, z):
            return np.where(np.abs(np.asarray(z, 'd') + 100.0) < 10.0, 50.0, 0.0)

        line = follow_terrain([(0, 0), (0, -200)], hill, spacing=5.0, smoothing=0.0,
                              maximum_grade=0.1)
        grades = np.abs(np.diff(line[:, 1])) / 5.0
        assert grades.max() <= 0.1 + 1e-9


class TestConformingTheGround:
    def _conformed(self, **kwargs):
        path = RoadPath(_straight(200.0, height=5.0), RoadProfile())
        return path, conform_terrain(_bumpy, path, **kwargs)

    def test_the_ground_under_the_road_is_the_road(self) -> None:
        """Less the formation the road is built on -- see
        :class:`TestTheGroundSitsUnderTheRoadNotInIt`."""
        path, ground = self._conformed(formation=0.0)
        height = float(ground(np.array([0.0]), np.array([-100.0]))[0])
        assert height == pytest.approx(5.0, abs=1e-6)

    def test_the_ground_takes_the_road_s_cross_section(self) -> None:
        """At the verge the ground is as far below the crown as the road is."""
        path, ground = self._conformed(formation=0.0)
        edge = path.profile.total_width / 2.0
        height = float(ground(np.array([edge]), np.array([-100.0]))[0])
        expected = 5.0 + float(path.section_offset(edge))
        assert height == pytest.approx(expected, abs=1e-6)

    def test_far_from_the_road_the_ground_is_untouched(self) -> None:
        _, ground = self._conformed()
        x, z = np.array([200.0]), np.array([-100.0])
        assert float(ground(x, z)[0]) == pytest.approx(float(_bumpy(x, z)[0]))

    def test_the_earthwork_has_no_step_in_it(self) -> None:
        path, ground = self._conformed()
        x = np.linspace(0.0, 60.0, 400)
        z = np.full_like(x, -100.0)
        heights = ground(x, z)
        steps = np.abs(np.diff(heights))
        assert steps.max() < 0.6, "a ridge where the earthwork meets the ground"

    def test_it_answers_a_grid_the_shape_it_was_asked(self) -> None:
        _, ground = self._conformed()
        x, z = np.meshgrid(np.linspace(-30, 30, 7), np.linspace(-150, -50, 5))
        assert np.asarray(ground(x, z)).shape == x.shape

    def test_a_tile_the_road_misses_costs_nothing_and_changes_nothing(self) -> None:
        _, ground = self._conformed()
        x, z = np.meshgrid(np.linspace(900, 1000, 9), np.linspace(900, 1000, 9))
        assert np.allclose(ground(x, z), _bumpy(x, z))


class TestTheRoadAsALayer:
    def _layer(self, **kwargs):
        return RoadLayer(RoadPath(_straight(400.0, count=41, height=3.0)), **kwargs)

    def test_a_tile_the_road_crosses_gets_it(self) -> None:
        region = BoundingBox((-50, -50, -300), (50, 50, -100))
        assert self._layer().content(region, error=0.0)

    def test_a_tile_the_road_misses_does_not(self) -> None:
        region = BoundingBox((500, -50, -300), (600, 50, -100))
        assert self._layer().content(region, error=0.0) == []

    def test_the_road_it_writes_is_inside_the_tile(self) -> None:
        region = BoundingBox((-50, -50, -300), (50, 50, -100))
        node = self._layer().content(region, error=0.0)[0]
        margin = self._layer().path.profile.total_width * 3
        assert node.mesh.positions[:, 2].min() >= -300 - margin
        assert node.mesh.positions[:, 2].max() <= -100 + margin

    def test_a_coarse_tile_spends_fewer_vertices(self) -> None:
        region = BoundingBox((-50, -50, -400), (50, 50, 0))
        fine = self._layer().content(region, error=0.0)[0]
        coarse = self._layer().content(region, error=64.0)[0]
        assert len(coarse.mesh.positions) < len(fine.mesh.positions) / 2

    def test_every_tile_gets_the_same_road_material(self) -> None:
        layer = self._layer()
        region = BoundingBox((-50, -50, -300), (50, 50, -100))
        first = layer.content(region, error=0.0)[0].mesh.material
        second = layer.content(region, error=8.0)[0].mesh.material
        assert first is second

    def test_a_road_entering_a_tile_twice_is_written_twice(self) -> None:
        """A hairpin crosses one tile going out and coming back."""
        out = np.stack([np.linspace(0, 100, 21), np.zeros(21),
                        np.full(21, -10.0)], axis=-1)
        back = np.stack([np.linspace(100, 0, 21), np.zeros(21),
                         np.full(21, -400.0)], axis=-1)
        turn = np.stack([np.full(11, 100.0), np.zeros(11),
                         np.linspace(-10, -400, 11)], axis=-1)
        layer = RoadLayer(RoadPath(np.vstack([out, turn, back])))
        region = BoundingBox((-10, -50, -420), (40, 50, 10))
        assert len(layer.content(region, error=0.0)) == 2

    def test_the_layer_covers_the_road(self) -> None:
        assert self._layer().bounds().minimum[2] <= -400

    def test_the_material_can_be_supplied(self) -> None:
        from OpenGLContext.scenegraph.road import tarmac_material
        wet = tarmac_material(wetness=0.8)
        region = BoundingBox((-50, -50, -300), (50, 50, -100))
        node = self._layer(material=wet).content(region, error=0.0)[0]
        assert node.mesh.material is wet


class TestACircuit:
    def _oval(self, count=64, radius=200.0):
        angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
        return np.stack([radius * np.cos(angle), radius * 0.6 * np.sin(angle)],
                        axis=-1)

    def test_it_closes(self) -> None:
        line = follow_terrain(self._oval(), _bumpy, spacing=8.0, closed=True)
        assert np.allclose(line[0], line[-1])

    def test_the_join_is_no_rougher_than_the_rest_of_the_lap(self) -> None:
        """A step at the start line would launch the car every lap."""
        line = follow_terrain(self._oval(), _bumpy, spacing=8.0, smoothing=80.0,
                              closed=True)
        steps = np.abs(np.diff(line[:, 1]))
        join = abs(line[1, 1] - line[-2, 1]) / 2.0
        assert join <= steps.max() + 1e-9

    def test_an_open_road_is_left_open(self) -> None:
        line = follow_terrain([(0, 0), (0, -200)], _bumpy, spacing=8.0)
        assert not np.allclose(line[0], line[-1])


class TestCarvingForACoarseTile:
    """A cutting narrower than the ground's sample spacing is stepped over.

    Widening the earthwork by the spacing puts a sample inside the corridor
    whatever the resolution, so the road is legible at every level of the tree
    instead of vanishing under the hillside it was cut into at all but the
    finest.
    """

    def _path(self):
        # A road ten metres below the natural ground: a cutting.
        return RoadPath(_straight(200.0, height=-10.0), RoadProfile())

    def test_a_widened_carve_reaches_further_out(self) -> None:
        path = self._path()
        narrow = conform_terrain(_flat, path, widening=0.0)
        wide = conform_terrain(_flat, path, widening=20.0)
        x, z = np.array([30.0]), np.array([-100.0])
        assert float(wide(x, z)[0]) < float(narrow(x, z)[0])

    def test_the_road_itself_is_still_at_road_level(self) -> None:
        wide = conform_terrain(_flat, self._path(), widening=20.0, formation=0.0)
        assert float(wide(np.array([0.0]), np.array([-100.0]))[0]) == pytest.approx(-10.0)

    def test_a_coarse_sample_beside_the_road_is_pulled_into_the_cutting(self) -> None:
        """The whole point: at 16 m spacing, the sample next to the road must
        already be down at road level, or the triangle between them roofs the
        road over."""
        spacing = 16.0
        ground = conform_terrain(_flat, self._path(), widening=spacing)
        half = self._path().profile.total_width / 2.0
        beside = float(ground(np.array([half + spacing]), np.array([-100.0]))[0])
        assert beside <= -10.0 + 0.6

    def test_far_from_the_road_a_wide_carve_still_leaves_the_ground_alone(self) -> None:
        ground = conform_terrain(_flat, self._path(), widening=16.0)
        assert float(ground(np.array([500.0]), np.array([-100.0]))[0]) == 0.0

    def test_the_factory_gives_a_function_per_spacing(self) -> None:
        from OpenGLContext_editor.world.road import conform_terrain_at
        at = conform_terrain_at(_flat, self._path())
        x, z = np.array([30.0]), np.array([-100.0])
        assert float(at(20.0)(x, z)[0]) < float(at(0.0)(x, z)[0])


class TestAWidenedCarveStillMeetsTheRoad:
    def test_it_never_rises_above_the_verge(self) -> None:
        """Widening moves the blend outwards; it must not lift the ground beside
        the road up to the crown, which would bury the shoulder."""
        path = RoadPath(_straight(200.0, height=-10.0), RoadProfile())
        half = path.profile.total_width / 2.0
        verge_level = -10.0 + float(path.section_offset(half))
        widening = 24.0
        ground = conform_terrain(_flat, path, widening=widening)
        # Out to the widening, the ground is held at the verge; beyond it the
        # blend climbs back to the natural hillside, which is the point of a
        # cutting.
        x = np.linspace(half, half + widening, 60)
        z = np.full_like(x, -100.0)
        assert np.all(np.asarray(ground(x, z)) <= verge_level + 1e-6)

    def test_the_cross_section_itself_is_untouched(self) -> None:
        path = RoadPath(_straight(200.0, height=5.0), RoadProfile())
        exact = conform_terrain(_flat, path, widening=0.0)
        wide = conform_terrain(_flat, path, widening=24.0)
        x = np.linspace(-path.profile.total_width / 2.0,
                        path.profile.total_width / 2.0, 40)
        z = np.full_like(x, -100.0)
        assert np.allclose(exact(x, z), wide(x, z))


class TestTheGroundNeverCoversTheRoad:
    """The invariant that matters on screen, checked on the meshed ground.

    A height function that agrees with the road proves nothing on its own: what
    a viewer sees is the straight lines a tile's mesh draws between its samples,
    and at a coarse spacing those can pass straight over a cutting. So the
    ground is actually meshed, at every spacing the tree will use, and the road
    is looked for underneath it.
    """

    def _road_over_hills(self):
        def hills(x, z):
            x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
            return 18.0 * np.sin(z * 0.012) + 9.0 * np.cos(x * 0.02)

        line = follow_terrain([(0, 0), (0, -600)], hills, spacing=6.0,
                              smoothing=60.0, maximum_grade=0.08)
        return hills, RoadPath(line, RoadProfile())

    @pytest.mark.parametrize('spacing', [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    def test_the_meshed_ground_stays_below_the_carriageway(self, spacing,
                                                           mesh_surface) -> None:
        from OpenGLContext.loaders.tiles3d.procedural import terrain_patch

        from OpenGLContext_editor.world.road import conform_terrain_at
        hills, path = self._road_over_hills()
        ground = conform_terrain_at(hills, path)(spacing)
        resolution = 65
        span = spacing * (resolution - 1)
        for centre in path.points[::20]:
            positions, _, _, indices = terrain_patch(
                centre[0] - span / 2, centre[0] + span / 2,
                centre[2] - span / 2, centre[2] + span / 2,
                resolution, height_fn=ground, water_level=None)
            height = mesh_surface(positions, indices, centre[0], centre[2])
            assert height is not None
            assert height <= centre[1] + 1e-3, (
                "the ground is %.2f m above the road at %s, spacing %s"
                % (height - centre[1], np.round(centre, 1), spacing))


class TestTilesDivideTheRoadBetweenThem:
    """Each stretch of road is written by exactly one tile.

    Two tiles both drawing the same stretch put two coplanar surfaces in the
    same place, and the join shows as a band fighting for the depth buffer --
    worse than the gap the overlap was meant to close.
    """

    def _quarters(self, size=400.0):
        """Four tiles side by side along the road's line."""
        edges = np.linspace(-size, size, 5)
        return [BoundingBox((-size, -500, edges[i], ), (size, 500, edges[i + 1]))
                for i in range(4)]

    def _layer(self):
        line = np.stack([np.zeros(201), np.full(201, 5.0),
                         np.linspace(-400.0, 400.0, 201)], axis=-1)
        return RoadLayer(RoadPath(line), finest_spacing=4.0)

    def test_the_tiles_between_them_cover_the_whole_road(self) -> None:
        layer = self._layer()
        covered = []
        for region in self._quarters():
            for node in layer.content(region, error=0.0):
                covered.append(node.mesh.positions[:, 2])
        span = np.concatenate(covered)
        assert span.min() <= -399.0 and span.max() >= 399.0

    def test_no_stretch_is_written_twice(self) -> None:
        layer = self._layer()
        spans = []
        for region in self._quarters():
            for node in layer.content(region, error=0.0):
                z = node.mesh.positions[:, 2]
                spans.append((float(z.min()), float(z.max())))
        spans.sort()
        for (_, first_end), (second_start, _) in pairwise(spans):
            # Neighbours share a vertex ring and overlap in nothing more.
            assert second_start >= first_end - 1e-6

    def test_neighbouring_tiles_meet_exactly(self) -> None:
        """A shared vertex ring, so there is no gap to see through either."""
        layer = self._layer()
        ends = []
        for region in self._quarters():
            for node in layer.content(region, error=0.0):
                z = node.mesh.positions[:, 2]
                ends.append((float(z.min()), float(z.max())))
        ends.sort()
        for (_, first_end), (second_start, _) in pairwise(ends):
            assert second_start == pytest.approx(first_end, abs=1e-6)


class TestCrossingLowGround:
    """A road does not run under water: it rides over on fill."""

    def _valley(self, x, z):
        """Ground that drops into a flooded basin in the middle."""
        z = np.asarray(z, 'd')
        return np.where(np.abs(z + 100.0) < 40.0, -12.0, 20.0)

    def test_the_alignment_stays_above_the_floor(self) -> None:
        line = follow_terrain([(0, 0), (0, -200)], self._valley, spacing=5.0,
                              smoothing=0.0, maximum_grade=0.08,
                              minimum_height=2.5)
        assert line[:, 1].min() >= 2.5 - 1e-9

    def test_the_approaches_climb_to_meet_it(self) -> None:
        """The crossing is not dropped back into the water to save the grade."""
        line = follow_terrain([(0, 0), (0, -200)], self._valley, spacing=5.0,
                              smoothing=0.0, maximum_grade=0.08,
                              minimum_height=2.5)
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        assert (np.abs(np.diff(line[:, 1])) / steps).max() <= 0.08 + 1e-9
        crossing = np.abs(line[:, 2] + 100.0) < 40.0
        assert line[crossing, 1].min() >= 2.5 - 1e-9

    def test_dry_ground_is_left_where_it_is(self) -> None:
        line = follow_terrain([(0, 0), (0, -200)], _flat, spacing=5.0,
                              smoothing=0.0, minimum_height=-50.0)
        assert np.allclose(line[:, 1], 0.0)


class TestTheWholeBakedWorldKeepsItsRoad:
    """The end-to-end statement: bake the shipped world and look under the road.

    Everything above tests a piece. This bakes the example world as a user
    would, walks every tile of every level, and asserts that nowhere in the tree
    does the ground it wrote cover the road it wrote. It is the check that would
    have caught each of the three separate ways this went wrong: a cutting too
    narrow for a coarse tile to resolve, a climbing road crossing its own
    earthwork between samples, and a circuit routed under a lake.
    """

    def test_no_tile_covers_the_road(self, tmp_path, mesh_surface) -> None:
        import numpy as np
        from OpenGLContext.loaders import gltf

        from OpenGLContext_editor.bake.driver import bake_world
        from OpenGLContext_editor.world.procedural import ProceduralWorld

        world = ProceduralWorld(extent=1024.0, resolution=17, seed=11)
        result = bake_world(world.layers(), str(tmp_path), depth=2)
        # Only where the road is laid on the land: over a bore the ground
        # stands above the carriageway by the whole depth of the hill, which is
        # what a tunnel is.
        circuit = world.circuit()
        line = circuit.points[circuit.on_ground]
        worst = 0.0
        for tile in _every_tile(result.tileset):
            ground = _ground_mesh(gltf.load_gltf(tile))
            if ground is None:
                continue
            positions = np.asarray(ground.positions, 'd')
            low, high = positions.min(axis=0), positions.max(axis=0)
            here = line[(line[:, 0] >= low[0]) & (line[:, 0] <= high[0])
                        & (line[:, 2] >= low[2]) & (line[:, 2] <= high[2])]
            for point in here[::5]:
                height = mesh_surface(positions, np.asarray(ground.indices),
                                      point[0], point[2])
                if height is not None:
                    worst = max(worst, height - point[1])
        assert worst <= 0.05, "the ground stands %.2f m over the road" % worst


def _every_tile(tileset_path):
    import json
    import os
    with open(tileset_path) as handle:
        document = json.load(handle)
    base = os.path.dirname(tileset_path)

    def walk(entry):
        content = entry.get('content')
        if content:
            yield os.path.join(base, content['uri'])
        for child in entry.get('children', []):
            yield from walk(child)

    return list(walk(document['root']))


def _ground_mesh(scene):
    """A tile's terrain: the mesh with per-vertex colour rather than a texture."""
    stack = [scene.group]
    while stack:
        node = stack.pop()
        geometry = getattr(node, 'geometry', None)
        if geometry is not None and getattr(geometry, 'colors', None) is not None:
            return geometry
        stack.extend(getattr(node, 'children', None) or [])
    return None


class TestTheRoadTravelsWithTheWorld:
    """A game cannot find a road in a pile of triangles, so the bake says."""

    def _layer(self):
        line = follow_terrain([(0, 0), (0, -400)], _bumpy, spacing=5.0)
        return RoadLayer(RoadPath(line))

    def test_the_centreline_is_in_the_metadata(self) -> None:
        roads = self._layer().metadata()['roads']
        assert len(roads) == 1
        assert len(roads[0]['centreline']) > 10
        assert len(roads[0]['centreline'][0]) == 3

    def test_it_says_how_wide_the_road_is(self) -> None:
        road = self._layer().metadata()['roads'][0]
        assert road['carriagewayWidth'] == pytest.approx(RoadProfile().carriageway_width)
        assert road['totalWidth'] > road['carriagewayWidth']

    def test_it_says_how_long_the_road_is(self) -> None:
        """Along the ground, so a road over hills is longer than its plan."""
        length = self._layer().metadata()['roads'][0]['length']
        assert 400.0 <= length < 420.0

    def test_an_open_road_says_it_is_open(self) -> None:
        assert self._layer().metadata()['roads'][0]['closed'] is False

    def test_a_circuit_says_it_closes(self) -> None:
        angle = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
        plan = np.stack([200 * np.cos(angle), 120 * np.sin(angle)], axis=-1)
        line = follow_terrain(plan, _bumpy, spacing=8.0, closed=True)
        assert RoadLayer(RoadPath(line)).metadata()['roads'][0]['closed'] is True

    def test_the_line_follows_the_road_it_baked(self) -> None:
        layer = self._layer()
        line = np.array(layer.metadata()['roads'][0]['centreline'])
        distance, _ = layer.path.nearest(line[:, 0], line[:, 2])
        assert float(np.max(distance)) < 0.5


class TestACircuitIsLimitedAllTheWayRound:
    """A circuit's steepest place is as likely to be the join as anywhere.

    The start line is exactly where a car is put, and a grade limit that ran
    from one end of the array to the other left that one point unconstrained --
    a cliff at the start line, and a car that fell off it.
    """

    def _hilly(self, x, z):
        x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
        return 60.0 * np.sin(x * 0.004) + 40.0 * np.cos(z * 0.005)

    def _circuit(self, **kwargs):
        angle = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
        plan = np.stack([700 * np.cos(angle), 500 * np.sin(angle)], axis=-1)
        return follow_terrain(plan, self._hilly, spacing=6.0, smoothing=90.0,
                              maximum_grade=0.075, closed=True, **kwargs)

    def _grades(self, line):
        closed = np.vstack([line, line[:1]])
        steps = np.linalg.norm(np.diff(closed[:, [0, 2]], axis=0), axis=1)
        return np.abs(np.diff(closed[:, 1])) / np.where(steps > 0, steps, 1.0)

    def test_no_step_anywhere_exceeds_the_grade(self) -> None:
        assert self._grades(self._circuit()).max() <= 0.075 + 1e-6

    def test_the_join_is_no_steeper_than_the_rest(self) -> None:
        grades = self._grades(self._circuit())
        assert grades[-1] <= grades.max()
        assert grades[0] <= grades.max()

    def test_it_still_follows_the_landscape(self) -> None:
        """Limiting the grade must not flatten the circuit into a ring road."""
        line = self._circuit()
        assert line[:, 1].max() - line[:, 1].min() > 20.0

    def test_a_water_floor_is_still_honoured_all_the_way_round(self) -> None:
        line = self._circuit(minimum_height=30.0)
        assert line[:, 1].min() >= 30.0 - 1e-9
        assert self._grades(line).max() <= 0.075 + 1e-6

    def test_an_open_road_is_unaffected(self) -> None:
        line = follow_terrain([(0, 0), (0, -600)], self._hilly, spacing=6.0,
                              maximum_grade=0.05)
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        assert (np.abs(np.diff(line[:, 1])) / steps).max() <= 0.05 + 1e-9


class TestTheRoadDoesNotLaunchACar:
    """A grade limit says how steeply a road may climb. It says nothing about
    how *suddenly* that may change, and a road that goes from climbing at its
    limit to descending at its limit inside a few metres is a ramp: a car
    arriving at speed leaves the ground, because there is nothing under it.

    Real roads round a change of grade off over a vertical curve whose length
    the design speed sets. So does this one.
    """

    def _ridge(self, x, z):
        """A knife-edge ridge across the course: up one side, down the other."""
        x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
        return 40.0 - 0.9 * np.abs(z + 300.0)

    def _over_the_ridge(self, **kwargs):
        plan = np.stack([np.zeros(161), np.linspace(0.0, -600.0, 161)], axis=-1)
        return follow_terrain(plan, self._ridge, spacing=6.0, smoothing=0.0,
                              maximum_grade=0.075, **kwargs)

    def _curvature(self, line):
        """Change of grade per metre along the alignment."""
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        grade = np.diff(line[:, 1]) / np.where(steps > 0, steps, 1.0)
        span = 0.5 * (steps[:-1] + steps[1:])
        return np.abs(np.diff(grade)) / np.where(span > 0, span, 1.0)

    def test_the_alignment_has_a_curvature_limit(self) -> None:
        line = self._over_the_ridge(design_speed=40.0)
        assert self._curvature(line).max() <= _curvature_limit(40.0) + 1e-6

    def test_a_slower_design_speed_allows_a_sharper_crest(self) -> None:
        slow = self._curvature(self._over_the_ridge(design_speed=15.0)).max()
        fast = self._curvature(self._over_the_ridge(design_speed=45.0)).max()
        assert slow > fast

    def test_the_grade_limit_still_holds(self) -> None:
        line = self._over_the_ridge(design_speed=40.0)
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        grade = np.abs(np.diff(line[:, 1])) / steps
        assert grade.max() <= 0.075 + 1e-3

    def test_the_crest_is_still_a_crest(self) -> None:
        """Rounding it off must not level the hill."""
        line = self._over_the_ridge(design_speed=40.0)
        assert line[:, 1].max() - line[:, 1].min() > 10.0

    def test_a_car_at_the_design_speed_keeps_its_wheels_down(self) -> None:
        """The whole point, stated as the physics it comes from: following the
        road must not need more downward acceleration than gravity gives."""
        speed = 40.0
        line = self._over_the_ridge(design_speed=speed)
        needed = self._curvature(line).max() * speed * speed
        assert needed < 9.81

    def test_it_is_off_unless_asked_for(self) -> None:
        """A designer who wants the line they drew gets the line they drew."""
        line = self._over_the_ridge()
        assert self._curvature(line).max() > _curvature_limit(40.0)


class TestACircuitIsRoundedAllTheWayRound:
    def _hilly(self, x, z):
        x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
        return 60.0 * np.sin(x * 0.004) + 40.0 * np.cos(z * 0.005)

    def _circuit(self, **kwargs):
        angle = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
        plan = np.stack([700 * np.cos(angle), 500 * np.sin(angle)], axis=-1)
        return follow_terrain(plan, self._hilly, spacing=6.0, smoothing=90.0,
                              maximum_grade=0.075, closed=True, **kwargs)

    def _curvature(self, line):
        """Curvature all the way round, the join included.

        ``follow_terrain`` closes a circuit by repeating its first point, so the
        wrap is built from the line without that repeat.
        """
        loop = line[:-1] if np.allclose(line[0], line[-1]) else line
        wrapped = np.vstack([loop, loop[:2]])
        steps = np.linalg.norm(np.diff(wrapped[:, [0, 2]], axis=0), axis=1)
        grade = np.diff(wrapped[:, 1]) / steps
        span = 0.5 * (steps[:-1] + steps[1:])
        return np.abs(np.diff(grade)) / span

    def test_the_join_is_rounded_like_everywhere_else(self) -> None:
        curvature = self._curvature(self._circuit(design_speed=40.0))
        assert curvature.max() <= _curvature_limit(40.0) + 1e-6

    def test_the_grade_limit_survives_the_rounding(self) -> None:
        line = self._circuit(design_speed=40.0)
        loop = line[:-1] if np.allclose(line[0], line[-1]) else line
        wrapped = np.vstack([loop, loop[:1]])
        steps = np.linalg.norm(np.diff(wrapped[:, [0, 2]], axis=0), axis=1)
        grade = np.abs(np.diff(wrapped[:, 1])) / steps
        assert grade.max() <= 0.075 + 1e-3

    def test_a_water_floor_is_still_honoured(self) -> None:
        line = self._circuit(design_speed=40.0, minimum_height=30.0)
        assert line[:, 1].min() >= 30.0 - 1e-9


class TestTheShippedCircuitIsDrivable:
    """The world ``oglc-bake`` produces is the one a player drives, so the
    question its alignment has to answer is a driver's: at the speed it is
    built for, does the car stay on the road?
    """

    def _circuit(self):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        return ProceduralWorld(extent=2048.0, resolution=17, seed=11).circuit()

    def _profile(self, line):
        loop = line[:-1] if np.allclose(line[0], line[-1]) else line
        wrapped = np.vstack([loop, loop[:2]])
        steps = np.linalg.norm(np.diff(wrapped[:, [0, 2]], axis=0), axis=1)
        grade = np.diff(wrapped[:, 1]) / steps
        span = 0.5 * (steps[:-1] + steps[1:])
        return np.abs(grade[:-1]), np.abs(np.diff(grade)) / span

    def test_no_crest_lifts_the_car_off_the_road(self) -> None:
        from OpenGLContext_editor.world.procedural import CIRCUIT_DESIGN_SPEED
        from OpenGLContext_editor.world.road import CREST_WEIGHT_LOSS, GRAVITY
        _grade, curvature = self._profile(self._circuit().points)
        lift = curvature.max() * CIRCUIT_DESIGN_SPEED ** 2 / GRAVITY
        assert lift <= CREST_WEIGHT_LOSS + 1e-6

    def test_no_climb_is_steeper_than_the_grade_limit(self) -> None:
        from OpenGLContext_editor.world.procedural import CIRCUIT_MAX_GRADE
        grade, _curvature = self._profile(self._circuit().points)
        assert grade.max() <= CIRCUIT_MAX_GRADE + 1e-3

    def test_it_is_still_a_circuit_through_hills(self) -> None:
        heights = self._circuit().points[:, 1]
        assert heights.max() - heights.min() > 20.0


class TestTheEarthworkMeetsTheGroundOnASlope:
    """A road that is not on the ground is on an earthwork, and an earthwork is
    a slope: fill runs down from the shoulder to where it meets the land, and a
    cutting runs up from it. How far out that is depends on how far the road is
    from the ground, and on nothing else.

    Returning to natural over a fixed distance instead leaves a road on a narrow
    shelf with the land falling away beside it -- correct at the road, a cliff
    two vehicle-widths out, and nothing a machine could have built.
    """

    def _across(self, ground, offsets, at=-100.0):
        x = np.asarray(offsets, 'd')
        return np.asarray(ground(x, np.full_like(x, at)), 'd')

    def _on_fill(self, height=40.0, **kwargs):
        """A road held well above flat ground: an embankment."""
        path = RoadPath(_straight(400.0, count=41, height=height), RoadProfile())
        return path, conform_terrain(_flat, path, **kwargs)

    def _in_cutting(self, depth=40.0, **kwargs):
        path = RoadPath(_straight(400.0, count=41, height=-depth), RoadProfile())
        return path, conform_terrain(_flat, path, **kwargs)

    def test_fill_runs_out_as_far_as_it_is_high(self) -> None:
        path, ground = self._on_fill(height=40.0, earthwork_slope=0.5)
        half = path.profile.total_width / 2.0
        # 40 m up at one in two is 80 m of batter; the ground is still raised
        # most of the way out and level again beyond.
        assert self._across(ground, [half + 40.0])[0] > 15.0
        assert self._across(ground, [half + 90.0])[0] == pytest.approx(0.0)

    def test_a_shallower_batter_reaches_further(self) -> None:
        _path, steep = self._on_fill(height=40.0, earthwork_slope=1.0)
        _path, shallow = self._on_fill(height=40.0, earthwork_slope=0.25)
        at = 60.0
        assert self._across(shallow, [at])[0] > self._across(steep, [at])[0]

    def test_the_batter_never_exceeds_its_slope(self) -> None:
        _path, ground = self._on_fill(height=40.0, earthwork_slope=0.5)
        x = np.linspace(0.0, 140.0, 1401)
        heights = self._across(ground, x)
        fall = np.abs(np.diff(heights)) / np.diff(x)
        assert fall.max() <= 0.5 + 1e-6

    def test_a_cutting_runs_up_the_same_way(self) -> None:
        path, ground = self._in_cutting(depth=40.0, earthwork_slope=0.5)
        half = path.profile.total_width / 2.0
        assert self._across(ground, [half + 40.0])[0] < -15.0
        assert self._across(ground, [half + 90.0])[0] == pytest.approx(0.0)

    def test_a_road_on_the_ground_disturbs_almost_nothing(self) -> None:
        """The common case: the alignment is already where the land is."""
        path = RoadPath(_straight(400.0, count=41, height=0.0), RoadProfile())
        ground = conform_terrain(_flat, path, earthwork_slope=0.5)
        half = path.profile.total_width / 2.0
        assert self._across(ground, [half + 5.0])[0] == pytest.approx(0.0, abs=0.1)

    def test_the_road_s_own_cross_section_is_unchanged(self) -> None:
        path, ground = self._on_fill(height=40.0, earthwork_slope=0.5,
                                     formation=0.0)
        half = path.profile.total_width / 2.0
        x = np.linspace(-half, half, 41)
        expected = 40.0 + np.asarray(path.section_offset(np.abs(x)), 'd')
        assert np.allclose(self._across(ground, x), expected, atol=1e-6)

    def test_an_earthwork_too_big_to_build_stops_at_its_limit(self) -> None:
        """A departure the batter cannot reach the ground within is left as it
        is: that is where a bridge or a tunnel belongs, and pretending
        otherwise would move a mountain to hide the fact."""
        path, ground = self._on_fill(height=400.0, earthwork_slope=0.5,
                                     maximum_earthwork=60.0)
        half = path.profile.total_width / 2.0
        assert self._across(ground, [half + 70.0])[0] == pytest.approx(0.0)

    def test_widening_still_holds_a_shelf_at_the_verge(self) -> None:
        """The coarse-tile carve: a flat shelf a sample wide before the batter."""
        path, ground = self._in_cutting(depth=20.0, earthwork_slope=0.5,
                                        widening=16.0)
        half = path.profile.total_width / 2.0
        verge = -20.0 + float(path.section_offset(half))
        x = np.linspace(half, half + 16.0, 40)
        assert np.all(self._across(ground, x) <= verge + 1e-6)


class TestTheGroundSitsUnderTheRoadNotInIt:
    """A road is built on a formation and surfaced on top of it, so the ground
    beneath is not the tarmac. Two surfaces at exactly the same height also
    fight over which one is drawn, which shows as the ground flickering through
    the carriageway in a sawtooth along the grid the terrain is sampled on.
    """

    def _ground(self, **kwargs):
        path = RoadPath(_straight(200.0, height=5.0), RoadProfile())
        return path, conform_terrain(_flat, path, **kwargs)

    def test_the_ground_is_below_the_carriageway(self) -> None:
        path, ground = self._ground()
        x = np.linspace(-4.0, 4.0, 17)
        heights = np.asarray(ground(x, np.full_like(x, -100.0)), 'd')
        road = 5.0 + np.asarray(path.section_offset(np.abs(x)), 'd')
        assert np.all(heights < road - 1e-6)

    def test_it_is_below_by_the_formation_depth(self) -> None:
        path, ground = self._ground(formation=0.4)
        height = float(ground(np.array([0.0]), np.array([-100.0]))[0])
        assert height == pytest.approx(5.0 - 0.4, abs=1e-6)

    def test_it_is_below_at_the_verge_too(self) -> None:
        """Or the ground would poke through where the batter starts."""
        path, ground = self._ground(formation=0.4)
        half = path.profile.total_width / 2.0
        height = float(ground(np.array([half]), np.array([-100.0]))[0])
        expected = 5.0 + float(path.section_offset(half)) - 0.4
        assert height == pytest.approx(expected, abs=1e-6)

    def test_the_earthwork_still_reaches_the_land(self) -> None:
        path, ground = self._ground(formation=0.4, earthwork_slope=0.5)
        assert float(ground(np.array([90.0]), np.array([-100.0]))[0]) \
            == pytest.approx(0.0)

    def test_it_is_shallow_enough_not_to_be_a_kerb(self) -> None:
        from OpenGLContext_editor.world.road import FORMATION_DEPTH
        assert 0.0 < FORMATION_DEPTH <= 0.25


class TestAWorldCanBeGivenItsOwnRoute:
    """The shipped world draws its own circuit. An editor draws one for it, and
    everything else about assembling a world -- the order the ground and the
    trees are settled in, the corridor kept clear, the credits -- is the same
    either way, so it is the same code either way.
    """

    def _plan(self, radius=400.0, points=64):
        angle = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
        return np.stack([radius * np.cos(angle), radius * 0.7 * np.sin(angle)],
                        axis=-1)

    def test_the_route_it_is_given_is_the_one_it_builds(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=2048.0, resolution=17,
                                route=self._plan(radius=400.0))
        circuit = world.circuit()
        radius = np.linalg.norm(circuit.points[:, [0, 2]], axis=1)
        assert radius.max() < 460.0
        assert radius.min() > 250.0

    def test_a_different_route_makes_a_different_circuit(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        small = ProceduralWorld(extent=2048.0, resolution=17,
                                route=self._plan(radius=200.0)).circuit()
        large = ProceduralWorld(extent=2048.0, resolution=17,
                                route=self._plan(radius=600.0)).circuit()
        assert large.length > small.length * 2.0

    def test_without_one_it_draws_its_own(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        assert ProceduralWorld(extent=2048.0, resolution=17).circuit().length > 0

    def test_the_route_is_still_settled_onto_the_ground(self) -> None:
        """A drawn route is a plan, not an alignment: it arrives with no
        heights on it and leaves with the grade limit honoured."""
        from OpenGLContext_editor.world.procedural import (
            CIRCUIT_MAX_GRADE,
            ProceduralWorld,
        )
        world = ProceduralWorld(extent=2048.0, resolution=17,
                                route=self._plan(radius=500.0))
        line = world.circuit().points
        steps = np.linalg.norm(np.diff(line[:, [0, 2]], axis=0), axis=1)
        grade = np.abs(np.diff(line[:, 1])) / np.where(steps > 0, steps, 1.0)
        assert grade.max() <= CIRCUIT_MAX_GRADE + 1e-3

    def test_an_open_route_is_a_road_rather_than_a_circuit(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        plan = np.stack([np.linspace(-800.0, 800.0, 40), np.zeros(40)], axis=-1)
        world = ProceduralWorld(extent=2048.0, resolution=17, route=plan,
                                closed=False)
        line = world.circuit().points
        assert not np.allclose(line[0], line[-1])


class TestFindingTheRoadQuickly:
    """Every ground sample in a world asks the road how far away it is, and the
    earthwork's reach is hundreds of metres, so the question is asked over a
    long line and answered a great many times. Comparing every sample against
    every segment is the whole cost of conforming a landscape.
    """

    def _circuit(self, points=600, radius=800.0):
        angle = np.linspace(0.0, 2 * np.pi, points, endpoint=False)
        line = np.stack([radius * np.cos(angle),
                         np.zeros(points),
                         radius * 0.7 * np.sin(angle)], axis=-1)
        return RoadPath(np.vstack([line, line[:1]]))

    def _grid(self, extent=2048.0, resolution=65):
        axis = np.linspace(-extent / 2, extent / 2, resolution)
        return np.meshgrid(axis, axis, indexing='ij')

    def test_it_answers_what_the_plain_search_answers(self) -> None:
        path = self._circuit()
        x, z = self._grid(resolution=33)
        quick = path.nearest(x, z, radius=260.0)
        every = path.nearest(x, z, radius=None)
        # Only where the plain search found something inside the radius: the
        # bounded one is allowed to say "further than that" and nothing more.
        near = quick[0] < 260.0
        assert np.allclose(quick[0][near], every[0][near], atol=1e-9)
        assert np.allclose(quick[1][near], every[1][near], atol=1e-9)

    def test_a_sample_far_from_the_road_is_out_of_reach(self) -> None:
        path = self._circuit()
        distance, _height = path.nearest(np.array([0.0]), np.array([0.0]),
                                         radius=100.0)
        assert not np.isfinite(distance[0])

    def test_it_does_not_compare_every_sample_with_every_segment(self) -> None:
        """The measurement that matters: the work, not the clock."""
        path = self._circuit(points=600)
        x, z = self._grid(resolution=65)
        compared = path.comparisons(x, z, radius=260.0)
        assert compared < 0.2 * x.size * 600

    def test_a_grid_nowhere_near_the_road_costs_almost_nothing(self) -> None:
        path = self._circuit(radius=200.0)
        axis = np.linspace(4000.0, 5000.0, 33)
        x, z = np.meshgrid(axis, axis, indexing='ij')
        assert path.comparisons(x, z, radius=100.0) == 0

    def test_the_shape_of_the_answer_still_matches_the_question(self) -> None:
        path = self._circuit()
        x, z = self._grid(resolution=17)
        distance, height = path.nearest(x, z, radius=260.0)
        assert distance.shape == x.shape and height.shape == x.shape

    def test_one_point_is_still_one_answer(self) -> None:
        path = self._circuit()
        distance, height = path.nearest(np.array([800.0]), np.array([0.0]),
                                        radius=260.0)
        assert distance.shape == (1,) and height.shape == (1,)
