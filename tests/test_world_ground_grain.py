"""The grain in a tiled world's ground, and what it is kept out of.

A world whose ground is meshed into its tiles adds micro-relief as the tree
refines, so standing in a tile buys hummocks and ruts rather than a denser
sampling of the same smooth function. What it must not reach is the landscape
beside the tileset: that is the surface a car is driven on, a camera is clamped
to and a tree is planted on, and it stays the analytic one.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph.roadworks import TunnelProfile
from OpenGLContext.scenegraph.terrain import GROUND_RELIEF, Relief

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.procedural import ProceduralWorld

EXTENT = 512.0


def _world(**kwargs):
    kwargs.setdefault('field_resolution', 129)
    return ProceduralWorld(extent=EXTENT, road=False, tree_density=0.0,
                           control_size=128, **kwargs)


def _tile(side=64.0):
    half = side / 2.0
    return BoundingBox((-half, -2000.0, -half), (half, 2000.0, half))


def _drawn(world, side=64.0, error=1.0):
    """The tile's *surface* vertices -- the skirt hangs below it by design."""
    layer = world.terrain()
    nodes = layer.content(_tile(side), error)
    across = layer.resolution ** 2
    return np.vstack([node.mesh.positions[:across] for node in nodes])


class TestTheGroundATiledWorldDraws:
    def test_it_carries_the_world_s_grain(self) -> None:
        world = _world(ground='tiles')
        assert world.terrain().relief == world.grain_drawn()
        assert world.grain_drawn().roughness == GROUND_RELIEF.roughness

    def test_and_a_world_may_ask_for_its_own(self) -> None:
        mine = Relief(coarsest=6.0, finest=1.5, roughness=0.04,
                      samples_per_feature=4.0)
        world = _world(ground='tiles', grain=mine, field_resolution=1537)
        assert world.terrain().relief == mine

    def test_but_none_of_it_finer_than_the_landscape_can_hold(self) -> None:
        """A band the collided surface cannot carry is relief a player sees and
        walks straight through, so nothing draws it."""
        world = _world(ground='tiles', field_resolution=513)
        drawn = world.terrain().relief
        assert len(drawn.wavelengths()) < len(GROUND_RELIEF.wavelengths())
        assert min(drawn.wavelengths()) \
            >= world.field_spacing() * drawn.samples_per_feature

    def test_and_the_landscape_is_built_with_the_same_grain(self) -> None:
        """The finest tile and the surface under it are one surface."""
        world = _world(ground='tiles')
        ground = world.landscape().height_fn
        smooth = world.height_fn()
        x = np.linspace(-100.0, 100.0, 41)
        z = np.full_like(x, 11.0)
        assert np.allclose(np.asarray(ground(x, z)) - np.asarray(smooth(x, z)),
                           world.grain_drawn().height(
                               x, z, spacing=world.ground_spacing(),
                               error=world.detail_error()))

    def test_or_for_none_at_all(self) -> None:
        world = _world(ground='tiles', grain=None)
        assert world.terrain().relief is None
        drawn = _drawn(world)
        smooth = world.height_fn()(drawn[:, 0], drawn[:, 2])
        assert np.allclose(drawn[:, 1], smooth, atol=1e-3)

    def test_a_close_tile_is_not_the_smooth_surface(self) -> None:
        world = _world(ground='tiles')
        drawn = _drawn(world, side=32.0, error=0.5)
        smooth = world.height_fn()(drawn[:, 0], drawn[:, 2])
        assert float(np.abs(drawn[:, 1] - smooth).max()) > 0.01

    def test_and_stands_no_further_off_than_the_tile_may_be_wrong_by(self) -> None:
        """Held to the tile's error, and never less than the finest tile's,
        which is the grain the landscape itself carries."""
        world = _world(ground='tiles')
        for error in (0.25, 1.0, 6.0):
            drawn = _drawn(world, side=32.0, error=error)
            smooth = world.height_fn()(drawn[:, 0], drawn[:, 2])
            held = max(error, world.detail_error())
            assert float(np.abs(drawn[:, 1] - smooth).max()) <= held + 1e-6


class TestTheFinestTile:
    """A bake gives its leaves an error of nought -- nothing finer follows --
    and the leaf is the tile drawn up close, over the landscape a car is
    driven on. It draws the grain that landscape carries."""

    def test_a_leaf_draws_the_surface_under_it(self) -> None:
        world = _world(ground='tiles')
        leaf = world.extent / 2 ** world.depth
        layer = world.terrain()
        drawn = layer.content(_tile(leaf), 0.0)[0].mesh.positions[
            :layer.resolution ** 2]
        under = world.detailed(world.height_fn())(drawn[:, 0], drawn[:, 2])
        smooth = world.height_fn()(drawn[:, 0], drawn[:, 2])
        assert np.abs(under - smooth).max() > 0.1          # there is grain
        assert np.allclose(drawn[:, 1], under, atol=1e-3)

    def test_a_coarser_tile_keeps_its_own_error(self) -> None:
        world = _world(ground='tiles')
        layer = world.terrain()
        error = world.detail_error() * 4.0
        side = world.extent / 2 ** (world.depth - 2)
        fn = layer.height_fn_for(_tile(side), error)
        x = np.linspace(-20.0, 20.0, 9)
        z = np.full_like(x, 3.0)
        assert np.allclose(fn(x, z), world.grain_drawn().over(
            world.height_fn(), spacing=layer.sample_spacing(_tile(side)),
            error=error)(x, z))


class TestWhatTheGrainIsKeptOutOf:
    def test_the_landscape_a_tiled_world_writes_carries_the_grain(self) -> None:
        """It is what the world is collided against, walked on and planted on.
        Relief drawn into a tile and left out of it is relief a player sees and
        walks straight through."""
        world = _world(ground='tiles')
        ground = world.landscape().field()
        x = np.linspace(-200.0, 200.0, 401)
        z = np.full_like(x, 37.0)
        moved = np.abs(np.asarray(ground.sample(x, z))
                       - np.asarray(world.height_fn()(x, z)))
        assert float(moved.max()) > 0.2
        assert float(moved.max()) <= world.detail_error() + 0.5

    def test_a_world_drawn_from_its_field_is_unmoved_by_the_grain(self) -> None:
        """A field world draws the landscape itself, so its ground is the
        analytic surface and the grain has nothing to attach to."""
        world = _world(ground='field')
        assert world.terrain() is world.landscape()
        assert world.grain_drawn() is None


class TestTheDefaultGrain:
    def test_a_world_has_one_without_asking(self) -> None:
        assert ProceduralWorld().grain is GROUND_RELIEF

    def test_and_a_refused_description_is_reported(self) -> None:
        with pytest.raises(ValueError):
            Relief(coarsest=0.5, finest=4.0)


class TestHowFinelyTheGroundIsMeshed:
    """A tiled world's ground is as fine as its *finest tile*, not as its root.
    Everything measured against a cell -- how wide a portal's face has to be to
    cover the hole cut for it, how far in front of one the road's own space is
    cleared -- is measured against that."""

    def test_a_field_world_is_as_fine_as_its_field(self) -> None:
        world = _world(ground='field', field_resolution=129)
        assert world.ground_spacing() == pytest.approx(EXTENT / 128.0)

    def test_a_tiled_world_is_as_fine_as_its_deepest_tile(self) -> None:
        world = _world(ground='tiles', resolution=33, depth=4)
        assert world.ground_spacing() == pytest.approx(EXTENT / 16.0 / 32.0)

    def test_so_a_deeper_tree_means_finer_ground(self) -> None:
        shallow = _world(ground='tiles', depth=2).ground_spacing()
        deep = _world(ground='tiles', depth=5).ground_spacing()
        assert deep < shallow

    def test_a_portal_face_covers_a_cell_of_it(self) -> None:
        """The hole is cut on the ground's own grid, so a face narrower than a
        cell leaves daylight down each side of the portal."""
        world = _world(ground='tiles', depth=4)
        assert world.tunnel_profile().portal_border \
            >= world.ground_spacing() - 1e-9

    def test_and_a_deep_tree_does_not_make_a_portal_out_of_one(self) -> None:
        """A face as wide as a root tile's cell is a headwall sixty metres
        across, which is a wall with a road-sized hole in it."""
        world = _world(ground='tiles', depth=5)
        assert world.tunnel_profile().portal_border \
            < 4.0 * TunnelProfile().portal_border


class TestTheGroundTheRoadWasBuiltOn:
    """A road is built by levelling the ground it runs on. Grain added there is
    ground standing up through the surface a car drives on -- which is what it
    was: the first tiled bake with grain in the landscape put the hillside
    1.57 m above the carriageway, and the car spent the run hitting it."""

    def _world(self):
        return ProceduralWorld(extent=1024.0, ground='tiles', depth=5,
                               tree_density=0.0, field_resolution=513,
                               control_size=128)

    def _on_the_line(self, world):
        path = world.circuit()
        points = path.points[::5]
        laid = np.asarray(path.segment_on_ground)[::5][:len(points)] > 0.5
        return points[laid]

    def test_the_carriageway_is_left_where_the_grader_left_it(self) -> None:
        world = self._world()
        points = self._on_the_line(world)
        smooth = np.asarray(world.height_fn()(points[:, 0], points[:, 2]))
        grained = np.asarray(world.detailed(world.height_fn())(
            points[:, 0], points[:, 2]))
        assert np.allclose(grained, smooth, atol=1e-6)

    def test_and_so_the_ground_still_meets_the_road_rather_than_covering_it(self) -> None:
        world = self._world()
        points = self._on_the_line(world)
        grained = np.asarray(world.detailed(world.height_fn())(
            points[:, 0], points[:, 2]))
        assert float((grained - points[:, 1]).max()) < 0.5

    def test_the_hillside_past_the_verge_keeps_its_grain(self) -> None:
        world = self._world()
        x = np.linspace(-400.0, 400.0, 401)
        z = np.full_like(x, 180.0)
        moved = np.asarray(world.detailed(world.height_fn())(x, z)) \
            - np.asarray(world.height_fn()(x, z))
        assert float(np.abs(moved).max()) > 0.2

    def test_and_it_comes_back_over_a_batter_rather_than_at_a_step(self) -> None:
        """Grain that switches on across one cell is a ridge down the length of
        the road."""
        world = self._world()
        applies = world.grain_applies()
        at = self._on_the_line(world)[0]
        side = np.array([0.0, 0.0, 1.0])
        across = np.asarray([at + side * step
                             for step in np.linspace(0.0, 40.0, 41)])
        weight = np.asarray(applies(across[:, 0], across[:, 2]))
        assert float(weight[0]) == pytest.approx(0.0, abs=1e-9)
        assert float(weight[-1]) == pytest.approx(1.0, abs=1e-9)
        assert float(np.max(np.diff(weight))) < 0.2

    def test_a_world_with_no_road_grades_nothing(self) -> None:
        world = ProceduralWorld(extent=512.0, ground='tiles', road=False,
                                tree_density=0.0, field_resolution=513)
        assert world.grain_applies() is None
