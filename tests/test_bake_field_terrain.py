"""Ground baked as one field rather than as tiles.

A tiled ground gets finer as the tree refines, which is what a world larger than
memory needs. A world of a few kilometres does not need it: the whole landscape
fits in one height field, and rendering it as a splat terrain -- one mesh, one
draw, detail materials blended per pixel -- costs less and looks better than a
tree of vertex-coloured patches.

So a world says which it wants. This is the field one: it contributes no tile
geometry at all, and instead writes the landscape beside the tileset as a
height image and a control map, with the numbers to read them back in the
tileset's ``extras``.
"""
import json
import math

import numpy as np
import pytest
from OpenGLContext.scenegraph.terrain import HeightField, LayerRule

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.field import FieldTerrainLayer

EXTENT = 1024.0


def _hilly(x, z):
    return 40.0 * np.sin(np.asarray(x, 'd') / 180.0) \
        - 30.0 * np.cos(np.asarray(z, 'd') / 220.0)


def _footprint():
    half = EXTENT / 2.0
    return BoundingBox((-half, 0.0, -half), (half, 0.0, half))


def _layer(**named):
    named.setdefault('height_fn', _hilly)
    named.setdefault('extent', _footprint())
    named.setdefault('resolution', 129)
    return FieldTerrainLayer(**named)


class TestItWritesTheLandscapeBesideTheTileset:
    def test_it_puts_no_geometry_in_a_tile(self) -> None:
        assert _layer().content(_footprint(), 4.0) == []

    def test_it_writes_a_height_image(self) -> None:
        assets = _layer().assets()
        assert any(name.endswith('.png') for name in assets)
        assert _layer().metadata()['terrain']['height'] in assets

    def test_it_writes_a_control_map(self) -> None:
        layer = _layer()
        assert layer.metadata()['terrain']['control'] in layer.assets()

    def test_the_images_are_real_files(self) -> None:
        import io

        from PIL import Image
        for raw in _layer().assets().values():
            assert Image.open(io.BytesIO(raw)).size[0] > 1

    def test_the_height_image_is_sixteen_bit(self) -> None:
        import io

        from PIL import Image
        layer = _layer()
        raw = layer.assets()[layer.metadata()['terrain']['height']]
        assert Image.open(io.BytesIO(raw)).mode in ('I', 'I;16', 'I;16B')

    def test_the_control_map_is_rgba(self) -> None:
        import io

        from PIL import Image
        layer = _layer()
        raw = layer.assets()[layer.metadata()['terrain']['control']]
        assert Image.open(io.BytesIO(raw)).mode == 'RGBA'

    def test_two_layers_of_one_world_do_not_share_a_filename(self) -> None:
        """Two terrains in one world would overwrite each other."""
        first = _layer(name='ground').assets()
        second = _layer(name='island').assets()
        assert not set(first) & set(second)


class TestWhatTheGameIsTold:
    def _terrain(self, **named):
        return _layer(**named).metadata()['terrain']

    def test_it_says_how_big_the_landscape_is(self) -> None:
        assert self._terrain()['extent'] == EXTENT

    def test_it_says_where_the_grid_stands_and_how_far_it_rises(self) -> None:
        found = self._terrain()
        assert found['base'] == pytest.approx(-70.0, abs=2.0)
        assert found['relief'] == pytest.approx(131.0, abs=5.0)

    def test_it_says_what_the_ground_is_made_of(self) -> None:
        assert self._terrain()['layers']

    def test_it_says_how_fine_the_grid_is(self) -> None:
        assert self._terrain()['resolution'] == 129

    def test_it_is_plain_json(self) -> None:
        assert json.loads(json.dumps(_layer().metadata()))

    def test_the_numbers_read_the_image_back_correctly(self, tmp_path) -> None:
        """The whole point of writing them: a game rebuilds the landscape the
        world was baked from, not one a scale-factor off."""
        layer = _layer()
        found = layer.metadata()['terrain']
        path = tmp_path / found['height']
        path.write_bytes(layer.assets()[found['height']])
        field = HeightField.from_image(str(path), found['resolution'],
                                       found['extent'], found['relief'],
                                       base=found['base'])
        x = np.linspace(-450.0, 450.0, 37)
        z = np.linspace(-400.0, 400.0, 37)
        assert np.abs(np.asarray(field.sample(x, z)) - _hilly(x, z)).max() < 2.0


class TestTheLandscapeItself:
    def test_the_field_it_builds_follows_the_height_function(self) -> None:
        field = _layer().field()
        x = np.linspace(-400.0, 400.0, 41)
        assert np.abs(np.asarray(field.sample(x, x)) - _hilly(x, x)).max() < 2.0

    def test_it_is_built_once_and_kept(self) -> None:
        """Sampling a conformed height function over a hundred thousand points
        is not something to do twice."""
        layer = _layer()
        assert layer.field() is layer.field()

    def test_the_bounds_hold_the_landscape(self) -> None:
        box = _layer().bounds()
        field = _layer().field()
        assert box.minimum[1] <= field.base
        assert box.maximum[1] >= field.base + field.relief

    def test_the_rules_decide_the_control_map(self) -> None:
        steep = _layer(rules=[LayerRule(), LayerRule(slope=(0.0, 1e-6))],
                       layers=['grass', 'rock'])
        assert len(steep.metadata()['terrain']['layers']) == 2


class TestPaintingTheRoadIn:
    def _straight_road(self):
        from OpenGLContext_editor.world.road import RoadPath
        x = np.linspace(-400.0, 400.0, 81)
        line = np.stack([x, _hilly(x, np.zeros_like(x)), np.zeros_like(x)],
                        axis=-1)
        return RoadPath(line)

    def test_the_corridor_takes_its_own_layer(self) -> None:
        """Gravel and bare earth beside a road, not grass up to the tarmac."""
        import io

        from PIL import Image
        layer = _layer(layers=['grass', 'forest_floor', 'rock', 'dirt'],
                       rules=[LayerRule(), LayerRule(), LayerRule(),
                              LayerRule()],
                       road=self._straight_road(), road_layer=3)
        raw = layer.assets()[layer.metadata()['terrain']['control']]
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), 'd') / 255.0
        size = pixels.shape[0]
        middle = size // 2
        on_road = pixels[middle, size // 2, 3]
        away = pixels[size // 4, size // 2, 3]
        assert on_road > away + 0.3

    def test_without_a_road_nothing_is_painted(self) -> None:
        import io

        from PIL import Image
        layer = _layer(layers=['grass', 'dirt'],
                       rules=[LayerRule(), LayerRule(slope=(9.0, 10.0))])
        raw = layer.assets()[layer.metadata()['terrain']['control']]
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), 'd') / 255.0
        assert pixels[..., 1].max() < 0.05


class TestBakedIntoAWorld:
    @pytest.fixture(scope='class')
    def baked(self, tmp_path_factory):
        from OpenGLContext_editor.bake.driver import bake_world
        directory = str(tmp_path_factory.mktemp('field'))
        layer = FieldTerrainLayer(height_fn=_hilly, extent=_footprint(),
                                  resolution=129)
        result = bake_world([layer], directory, depth=1)
        with open(result.tileset) as handle:
            return result, json.load(handle)

    def test_the_images_land_beside_the_tileset(self, baked) -> None:
        import os
        result, document = baked
        terrain = document['extras']['terrain']
        for key in ('height', 'control'):
            assert os.path.exists(os.path.join(result.directory, terrain[key]))

    def test_the_tileset_carries_the_numbers(self, baked) -> None:
        _result, document = baked
        assert document['extras']['terrain']['extent'] == EXTENT

    def test_the_world_still_has_a_bounding_volume(self, baked) -> None:
        """A tileset whose root covers nothing is a tileset a viewer never
        looks at."""
        _result, document = baked
        box = document['root']['boundingVolume']['box']
        assert max(abs(v) for v in box[3:]) > 0.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestGroundThatHasToCarryARoad:
    """A road's cutting is metres wide and the field's grid is metres apart, so
    the two have to be sampled against each other or the road is smoothed away
    and comes out buried in the hillside it was cut into."""

    def _world(self, **named):
        from OpenGLContext.loaders.tiles3d.procedural import terrain_height
        from OpenGLContext_editor.world.road import (
            RoadPath, conform_terrain, conform_terrain_at, follow_terrain,
        )
        plan = np.stack([np.linspace(-450.0, 450.0, 40),
                         np.linspace(-300.0, 300.0, 40)], axis=-1)
        # Heavily smoothed, so the alignment ignores the hummocks and the
        # ground has to be cut away to meet it -- which is the case a coarse
        # grid loses.
        line = follow_terrain(plan, terrain_height, spacing=6.0, smoothing=400.0,
                              maximum_grade=0.05)
        road = RoadPath(line)
        named.setdefault('height_fn', conform_terrain(terrain_height, road))
        named.setdefault('height_fn_at', conform_terrain_at(terrain_height, road))
        named.setdefault('extent', _footprint())
        named.setdefault('resolution', 257)
        named.setdefault('road', road)
        return FieldTerrainLayer(**named), road

    def test_the_ground_meets_the_road_it_carries(self) -> None:
        layer, road = self._world()
        field = layer.field()
        line = road.points
        over = np.asarray(field.sample(line[:, 0], line[:, 2])) - line[:, 1]
        assert float(over.max()) < 1.0, "the ground stands over the road"

    def test_it_holds_a_shelf_the_grid_can_see(self) -> None:
        """The point of ``height_fn_at``: a corridor narrower than the grid is
        stepped over, so the ground is held at the verge for a whole cell
        before the batter starts."""
        widened, road = self._world(resolution=65)
        plain, _road = self._world(height_fn_at=None, resolution=65)
        line = road.points
        step = np.diff(line[:, [0, 2]], axis=0, append=line[:1, [0, 2]])
        norm = np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
        out = (road.profile.total_width / 2.0 + 8.0) / norm[:, 0]
        beside_x = line[:, 0] - step[:, 1] * out
        beside_z = line[:, 2] + step[:, 0] * out
        near = np.abs(np.asarray(widened.field().sample(beside_x, beside_z))
                      - line[:, 1])
        far = np.abs(np.asarray(plain.field().sample(beside_x, beside_z))
                     - line[:, 1])
        assert float(np.median(near)) < float(np.median(far))

    def test_the_spacing_it_asks_for_is_its_own(self) -> None:
        asked = []

        def watching(spacing):
            asked.append(spacing)
            return lambda x, z: np.zeros(np.shape(x))
        layer, _road = self._world(height_fn_at=watching, resolution=129)
        layer.field()
        assert asked == [pytest.approx(EXTENT / 128.0)]

    def test_the_road_still_paints_its_corridor(self) -> None:
        import io

        from PIL import Image
        layer, _road = self._world()
        raw = layer.assets()[layer.metadata()['terrain']['control']]
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), 'd') / 255.0
        assert pixels[..., 3].max() > 0.9


class TestHowWideTheRoadsGroundIs:
    def _straight_road(self):
        from OpenGLContext_editor.world.road import RoadPath
        x = np.linspace(-400.0, 400.0, 81)
        line = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)
        return RoadPath(line)

    def _across(self, **named):
        import io

        from PIL import Image
        layer = _layer(height_fn=lambda x, z: np.zeros(np.shape(np.asarray(x))),
                       layers=['grass', 'dirt'],
                       rules=[LayerRule(), LayerRule(weight=0.0)],
                       road=self._straight_road(), road_layer=1, **named)
        raw = layer.assets()[layer.metadata()['terrain']['control']]
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGBA'), 'd') / 255.0
        size = pixels.shape[0]
        band = pixels[:, size // 2, 1]
        metres = EXTENT / (size - 1)
        return float((band > 0.5).sum()) * metres

    def test_a_corridor_in_metres_is_the_one_used(self) -> None:
        assert self._across(road_corridor=40.0) == pytest.approx(80.0, abs=12.0)

    def test_a_narrow_one_paints_a_narrow_strip(self) -> None:
        assert self._across(road_corridor=8.0) < self._across(road_corridor=40.0)

    def test_without_one_it_follows_the_road_s_width(self) -> None:
        assert self._across() > 0.0
