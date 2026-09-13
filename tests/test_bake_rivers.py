"""Rivers in a baked world, and what one looks like from far off.

A river carved into the terrain is a valley; what makes it a river is water in
it, and a world nobody can drive to the water of is a world with no rivers. The
level of detail matters as much: a full surface a kilometre away spends a
tile's whole budget on a line two pixels wide.
"""
import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.rivers import RiverLayer
from OpenGLContext_editor.world.hydrology import Channel


def _slope(x, z):
    return 200.0 - np.asarray(x, dtype='d') * 0.1


def _channel(length=800.0, points=41, flow=1.0):
    x = np.linspace(-length / 2.0, length / 2.0, points)
    z = np.zeros_like(x)
    return Channel(points=np.stack([x, z], axis=-1),
                   flow=np.full(points, float(flow)))


def _layer(**named):
    named.setdefault('channels', [_channel()])
    named.setdefault('ground', _slope)
    return RiverLayer(**named)


def _region(half=600.0, low=-500.0, high=500.0):
    return BoundingBox((-half, low, -half), (half, high, half))


class TestWhatIsInATile:
    def test_a_tile_the_river_crosses_holds_water(self) -> None:
        assert _layer().content(_region(), error=1.0)

    def test_a_tile_it_misses_holds_none(self) -> None:
        away = BoundingBox((4000.0, -500.0, 4000.0), (5000.0, 500.0, 5000.0))
        assert _layer().content(away, error=1.0) == []

    def test_a_world_with_no_rivers_writes_none(self) -> None:
        assert _layer(channels=[]).content(_region(), error=1.0) == []

    def test_the_water_is_where_the_channel_is(self) -> None:
        nodes = _layer().content(_region(), error=1.0)
        points = np.vstack([np.asarray(node.mesh.positions) for node in nodes])
        assert np.abs(points[:, 2]).max() < 40.0

    def test_it_runs_downhill(self) -> None:
        nodes = _layer().content(_region(), error=1.0)
        points = np.vstack([np.asarray(node.mesh.positions) for node in nodes])
        upstream = points[points[:, 0] < -300.0][:, 1].mean()
        downstream = points[points[:, 0] > 300.0][:, 1].mean()
        assert upstream > downstream

    def test_it_sits_in_the_bed_it_cut(self) -> None:
        """Below the land the channel was carved into, or it is a ribbon
        draped over a hillside."""
        nodes = _layer().content(_region(), error=1.0)
        points = np.vstack([np.asarray(node.mesh.positions) for node in nodes])
        land = np.asarray(_slope(points[:, 0], points[:, 2]), dtype='d')
        assert (points[:, 1] < land).mean() > 0.9


class TestFromFarOff:
    def test_a_close_tile_gets_the_whole_surface(self) -> None:
        near = _layer().content(_region(), error=0.5)
        far = _layer().content(_region(), error=200.0)
        assert sum(len(n.mesh.positions) for n in near) \
            > sum(len(n.mesh.positions) for n in far)

    def test_a_river_never_vanishes_however_far_off(self) -> None:
        """It may be one glint, but a river that is in the tile is drawn."""
        nodes = _layer().content(_region(), error=4000.0)
        assert nodes and all(len(n.mesh.positions) for n in nodes)

    def test_a_distant_tile_gets_glints_instead(self) -> None:
        nodes = _layer().content(_region(), error=200.0)
        assert nodes
        assert any('glint' in node.name for node in nodes)

    def test_a_close_tile_does_not(self) -> None:
        nodes = _layer().content(_region(), error=0.5)
        assert not any('glint' in node.name for node in nodes)

    def test_the_glints_still_lie_along_the_river(self) -> None:
        """At an error a tile this size would actually be drawn at."""
        nodes = _layer().content(_region(), error=30.0)
        points = np.vstack([np.asarray(node.mesh.positions) for node in nodes])
        assert np.abs(points[:, 2]).max() < 40.0
        assert np.ptp(points[:, 0]) > 300.0

    def test_further_off_still_means_fewer_of_them(self) -> None:
        """A coarser tile is also a bigger tile, so the spacing growing with
        the error is what keeps the count in a tile roughly constant."""
        some = _layer().content(_region(), error=30.0)
        fewer = _layer().content(_region(), error=200.0)
        assert sum(len(n.mesh.positions) for n in fewer) \
            <= sum(len(n.mesh.positions) for n in some)


class TestWhereItSaysitIs:
    def test_it_reports_the_ground_its_rivers_cover(self) -> None:
        box = _layer().bounds()
        assert box.minimum[0] <= -400.0 and box.maximum[0] >= 400.0

    def test_a_world_with_no_rivers_reports_nothing_to_bake(self) -> None:
        assert _layer(channels=[]).bounds() is None

    def test_the_box_has_the_water_in_it(self) -> None:
        box = _layer().bounds()
        nodes = _layer().content(_region(), error=1.0)
        points = np.vstack([np.asarray(node.mesh.positions) for node in nodes])
        assert box.minimum[1] <= points[:, 1].min() + 1e-6
        assert box.maximum[1] >= points[:, 1].max() - 1e-6


class TestTheWorldThatCarriesThem:
    """A world is a list of layers, and the rivers have to be among them."""

    def _world(self, channels=(), **named):
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        named.setdefault('extent', 1024.0)
        named.setdefault('resolution', 17)
        named.setdefault('tree_density', 0.0)
        named.setdefault('road', False)
        # The default forest reads the demo's tree files; rivers do not
        # need them, and `tree_density=0.0` above does not stand them
        # down -- the species list is resolved whether or not anything
        # stands on it.
        named.setdefault('forest', 'tiles')
        return ProceduralWorld(channels=list(channels), **named)

    def test_a_world_with_no_rivers_has_no_river_layer(self) -> None:
        names = [getattr(layer, 'name', '') for layer in self._world().layers()]
        assert 'river' not in names

    def test_a_world_with_rivers_carries_them(self) -> None:
        world = self._world(channels=[_channel(length=400.0)])
        names = [getattr(layer, 'name', '') for layer in world.layers()]
        assert 'river' in names

    def test_the_layer_is_given_the_land_the_beds_were_cut_into(self) -> None:
        """Not the ground with the channels in it, or the water would be
        measured down from the bed it is supposed to fill."""
        world = self._world(channels=[_channel(length=400.0)])
        rivers = [layer for layer in world.layers()
                  if getattr(layer, 'name', '') == 'river'][0]
        assert rivers.ground is not None
        x = np.asarray([0.0])
        z = np.asarray([0.0])
        assert float(np.asarray(rivers.ground(x, z))[0]) \
            == pytest.approx(float(np.asarray(world.natural()(x, z))[0]))

    def test_it_writes_water_into_a_tile(self) -> None:
        world = self._world(channels=[_channel(length=400.0)])
        rivers = [layer for layer in world.layers()
                  if getattr(layer, 'name', '') == 'river'][0]
        assert rivers.content(_region(half=300.0), error=1.0)
