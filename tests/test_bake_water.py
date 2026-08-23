"""Open water in a baked world: where it is, and where it is not.

Water is its own surface rather than the ground clamped flat, so the layer's
whole job is deciding *where*: a tile whose ground never reaches the waterline
holds no water, and one that dips below it is covered edge to edge, because a
lake does not stop halfway across a tile.
"""
import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.layers import WaterLayer

EXTENT = BoundingBox((-100.0, 0.0, -100.0), (100.0, 0.0, 100.0))


def _basin(x, z):
    """Ground dipping to -25 in the middle, rising past zero outside r=100."""
    x = np.asarray(x, dtype='d')
    z = np.asarray(z, dtype='d')
    return (x * x + z * z) / 400.0 - 25.0


def _hill(x, z):
    """Ground everywhere above the waterline."""
    return np.full(np.broadcast(np.asarray(x), np.asarray(z)).shape, 40.0)


def _layer(height_fn=_basin, **named):
    return WaterLayer(height_fn=height_fn, extent=EXTENT, **named)


def _over(box, layer, error=1.0):
    return layer.content(box, error)


class TestWhereThereIsWater:
    def test_a_basin_holds_some(self):
        region = BoundingBox((-50.0, -40.0, -50.0), (50.0, 40.0, 50.0))
        assert _over(region, _layer())

    def test_ground_that_never_reaches_the_line_holds_none(self):
        region = BoundingBox((-50.0, 0.0, -50.0), (50.0, 80.0, 50.0))
        assert _over(region, _layer(_hill)) == []

    def test_a_region_the_water_is_nowhere_near_holds_none(self):
        """A tile high in the air over a lake is still not a tile of lake."""
        region = BoundingBox((-50.0, 200.0, -50.0), (50.0, 300.0, 50.0))
        assert _over(region, _layer()) == []

    def test_a_region_that_misses_the_extent_holds_none(self):
        region = BoundingBox((500.0, -40.0, 500.0), (600.0, 40.0, 600.0))
        assert _over(region, _layer()) == []

    def test_a_raised_waterline_floods_more_of_it(self):
        """Ground between the old line and the new is under water now."""
        # Beyond the shoreline at level 0: the ground here stands at +7.
        corner = BoundingBox((80.0, -40.0, 80.0), (100.0, 40.0, 100.0))
        assert _over(corner, _layer(level=0.0)) == []
        assert _over(corner, _layer(level=12.0))


class TestWhatItPutsThere:
    pass
    def _sheet(self, **named):
        region = BoundingBox((-50.0, -40.0, -50.0), (50.0, 40.0, 50.0))
        return _over(region, _layer(**named))[0]

    def test_the_water_covers_the_whole_tile(self):
        """A lake does not stop halfway across one."""
        mesh = self._sheet().mesh
        assert (float(mesh.positions[:, 0].min()),
                float(mesh.positions[:, 0].max())) == pytest.approx((-50.0, 50.0))

    def test_and_lies_at_the_waterline(self):
        """Sits at it rather than being flat on it: a lake carries a swell, so
        what is level is the water, not each vertex of it."""
        from OpenGLContext.scenegraph.water import LAKE
        heights = self._sheet(level=3.5).mesh.positions[:, 1]
        assert float(np.mean(heights)) == pytest.approx(3.5, abs=0.05)
        # Three trains cross, so the excursion is a small multiple of one
        # train's own amplitude rather than that amplitude exactly.
        assert float(np.abs(heights - 3.5).max()) <= LAKE.amplitude * 2.5

    def test_and_it_is_not_a_flat_plate(self):
        """Still water is a mirror, and a mirror that size with nothing over it
        but a pale sky is a white plate lying in the landscape."""
        heights = self._sheet(level=3.5).mesh.positions[:, 1]
        assert float(heights.max() - heights.min()) > 0.05

    def test_it_is_clipped_to_the_layer_s_own_extent(self):
        """Water is not baked over ground the world does not have."""
        region = BoundingBox((-200.0, -40.0, -200.0), (0.0, 40.0, 0.0))
        mesh = _over(region, _layer())[0].mesh
        assert float(mesh.positions[:, 0].min()) == pytest.approx(-100.0)

    def test_the_node_says_what_it_is(self):
        assert 'water' in self._sheet().name

    def test_a_caller_may_name_it_something_else(self):
        assert 'lake' in self._sheet(name='lake').name


class TestHowFarItReaches:
    def test_the_layer_covers_its_extent(self):
        found = _layer().bounds()
        assert (float(found.minimum[0]), float(found.maximum[0])) == \
            pytest.approx((-100.0, 100.0))

    def test_and_sits_at_the_waterline(self):
        found = _layer(level=7.5).bounds()
        assert float(found.minimum[1]) == pytest.approx(7.5)
        assert float(found.maximum[1]) == pytest.approx(7.5)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
