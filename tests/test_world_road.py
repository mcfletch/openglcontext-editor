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
        path, ground = self._conformed()
        height = float(ground(np.array([0.0]), np.array([-100.0]))[0])
        assert height == pytest.approx(5.0, abs=1e-6)

    def test_the_ground_takes_the_road_s_cross_section(self) -> None:
        """At the verge the ground is as far below the crown as the road is."""
        path, ground = self._conformed()
        edge = path.profile.total_width / 2.0
        height = float(ground(np.array([edge]), np.array([-100.0]))[0])
        expected = 5.0 + float(path.section_offset(edge))
        assert height == pytest.approx(expected, abs=1e-6)

    def test_far_from_the_road_the_ground_is_untouched(self) -> None:
        _, ground = self._conformed(blend=10.0)
        x, z = np.array([200.0]), np.array([-100.0])
        assert float(ground(x, z)[0]) == pytest.approx(float(_bumpy(x, z)[0]))

    def test_the_earthwork_has_no_step_in_it(self) -> None:
        path, ground = self._conformed(blend=12.0)
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
        wide = conform_terrain(_flat, self._path(), widening=20.0)
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
        line = world.circuit().points
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
