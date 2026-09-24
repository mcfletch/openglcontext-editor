"""The places a road runs through, and the zones a world is baked with for them."""
import io
import json
import math
import wave

import numpy as np
import pytest

from OpenGLContext_editor.bake.zones import (
    DOCUMENT,
    AmbientSound,
    ZonesLayer,
    place_sounds,
    wav_bytes,
    zone_records,
)
from OpenGLContext_editor.world.places import (
    BRIDGE,
    CAUSEWAY,
    FOREST,
    KINDS,
    TUNNEL,
    road_places,
)
from OpenGLContext_editor.world.road import RoadPath
from OpenGLContext_editor.world.structures import Op


def straight(length=1000.0, count=201, ops=None, heading=0.0):
    """A straight road along ``heading`` (radians from +X towards +Z)."""
    t = np.linspace(0.0, length, count)
    points = np.stack([t * math.cos(heading), np.full(count, 10.0),
                       t * math.sin(heading)], axis=-1)
    return RoadPath(points, ops=ops)


def with_run(op, start, end, length=1000.0, count=201):
    stations = np.linspace(0.0, length, count)
    ops = [op if start <= s <= end else Op.DIRT for s in stations]
    return straight(length, count, ops)


def trees_beside(road, start, end, spacing=4.0, offset=12.0):
    """A row of trees each side of the road from ``start`` to ``end``."""
    s = np.arange(start, end, spacing)
    return np.concatenate([np.stack([s, np.zeros_like(s), np.full_like(s, side)], axis=-1)
                           for side in (-offset, offset)])


class TestPlaces:
    def test_a_bore_is_boxes_along_it(self):
        road = with_run(Op.TUNNEL, 300.0, 520.0)
        places = road_places(road, tunnel_half_width=6.0, tunnel_height=7.5)
        assert {p.kind for p in places} == {TUNNEL}
        assert len(places) == math.ceil(220.0 / KINDS[TUNNEL].chunk)
        assert min(p.start for p in places) == pytest.approx(300.0, abs=5.0)
        assert max(p.end for p in places) == pytest.approx(520.0, abs=5.0)

    def test_a_box_covers_its_stretch_of_road(self):
        road = with_run(Op.TUNNEL, 300.0, 380.0)
        (place,) = road_places(road, tunnel_half_width=6.0, tunnel_height=7.5)
        along, height, across = place.size
        assert along == pytest.approx(80.0, abs=5.0)
        assert across == pytest.approx(2 * (6.0 + KINDS[TUNNEL].across))
        assert place.centre[0] == pytest.approx(340.0, abs=3.0)
        assert place.centre[2] == pytest.approx(0.0, abs=1e-6)
        assert place.centre[1] - height / 2 < 10.0 < place.centre[1] + height / 2

    def test_the_box_turns_with_the_road(self):
        heading = math.radians(30.0)
        stations = np.linspace(0.0, 1000.0, 201)
        ops = [Op.CAUSEWAY if 200 <= s <= 300 else Op.DIRT for s in stations]
        road = straight(ops=ops, heading=heading)
        (place,) = road_places(road)
        x, y, z, w = place.rotation
        # The box's local +X, turned by its rotation, runs along the road.
        angle = 2.0 * math.atan2(y, w)
        local_x = (math.cos(angle), -math.sin(angle))
        assert local_x == pytest.approx((math.cos(heading), math.sin(heading)), abs=1e-6)

    def test_the_eye_is_on_the_road_above_it(self):
        road = with_run(Op.BRIDGE, 100.0, 200.0)
        (place,) = road_places(road)
        assert place.kind == BRIDGE
        assert place.centre[1] + place.eye[1] == pytest.approx(12.0)
        assert place.eye[2] == pytest.approx(0.0, abs=1e-6)

    def test_plain_road_through_trees_is_forest(self):
        road = straight()
        places = road_places(road, trees_beside(road, 0.0, 400.0))
        forest = [p for p in places if p.kind == FOREST]
        assert forest and max(p.end for p in forest) <= 400.0 + 1e-6

    def test_open_ground_has_no_place(self):
        road = straight()
        assert road_places(road, np.zeros((0, 3))) == []
        assert road_places(road, trees_beside(road, 0.0, 1000.0, spacing=200.0)) == []

    def test_a_structure_is_not_also_forest(self):
        road = with_run(Op.CAUSEWAY, 0.0, 1000.0)
        places = road_places(road, trees_beside(road, 0.0, 1000.0))
        assert {p.kind for p in places} == {CAUSEWAY}


class TestTheLayer:
    def _layer(self):
        road = with_run(Op.TUNNEL, 300.0, 380.0)
        places = road_places(road, trees_beside(road, 500.0, 900.0),
                             tunnel_half_width=6.0)
        sounds = {'birdsong': AmbientSound('birdsong', lambda: np.zeros(800), 0.5),
                  'surf': AmbientSound('surf', lambda: np.zeros(800), 0.5)}
        return ZonesLayer(zone_records(places), sounds)

    def test_it_puts_nothing_in_any_tile(self):
        layer = self._layer()
        assert layer.bounds() is None and layer.content(None, 0.0) == []

    def test_the_tileset_names_the_document(self):
        record = self._layer().metadata()['zones']
        assert record['document'] == DOCUMENT and record['count'] > 1

    def test_the_document_is_zones_the_engine_reads(self, tmp_path):
        from OpenGLContext.loaders.gltf import loader
        from OpenGLContext.scenegraph.zone import AUDIO, ENVIRONMENT, REVERB
        layer = self._layer()
        for name, data in layer.assets().items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        scene = loader.load_gltf(str(tmp_path / DOCUMENT))
        kinds = {zone.priority: zone for zone in scene.zones}
        bore, forest = kinds[KINDS[TUNNEL].priority], kinds[KINDS[FOREST].priority]
        assert bore.setting(REVERB) is not None and bore.setting(AUDIO) is None
        assert forest.setting(AUDIO) is not None
        assert all(zone.setting(ENVIRONMENT).capture for zone in scene.zones)
        assert set(scene.sounds) == {'birdsong'}

    def test_only_the_sounds_played_are_written(self):
        names = set(self._layer().assets())
        assert names == {DOCUMENT, 'audio/birdsong.wav'}

    def test_a_zone_playing_an_unknown_sound_is_refused(self):
        layer = self._layer()
        layer.sounds = {}
        with pytest.raises(ValueError):
            layer.assets()

    def test_a_world_with_no_places_writes_nothing(self):
        layer = ZonesLayer([], {})
        assert layer.metadata() == {} and layer.assets() == {}

    def test_the_document_is_glTF_2_0_with_its_shapes(self):
        doc = self._layer().document()
        assert doc['asset']['version'] == '2.0'
        assert {'OGLC_zone', 'KHR_implicit_shapes'} <= set(doc['extensionsUsed'])
        assert 'extensionsRequired' not in doc
        shapes = doc['extensions']['KHR_implicit_shapes']['shapes']
        assert len(shapes) == len(doc['nodes'])
        json.dumps(doc)


def test_the_places_sound_like_their_places():
    sounds = place_sounds(seed=3)
    assert set(sounds) == {'birdsong', 'surf'}
    with wave.open(io.BytesIO(wav_bytes(sounds['surf'].make()))) as handle:
        assert handle.getnchannels() == 1
        assert handle.getnframes() == handle.getframerate() * 14
