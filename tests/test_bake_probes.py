"""Baking zone environments into a world: the document the bake leaves behind."""
import io
import json
import os

import numpy as np
from PIL import Image

from OpenGLContext_editor.bake import probes
from OpenGLContext_editor.bake.zones import ZoneRecord, ZonesLayer


def _document(tmp_path, names=('bore-0', 'bore-1', 'bore-2')):
    zones = [ZoneRecord(name=name, centre=(i * 10.0, 0, 0),
                        rotation=(0, 0, 0, 1), size=(8, 8, 8),
                        environment={'capture': {'position': [0, 1, 0]}, 'intensity': 0.8})
             for i, name in enumerate(names)]
    path = tmp_path / 'zones.gltf'
    path.write_text(json.dumps(ZonesLayer(zones, {}).document()))
    return path


def _light(name, tag='a'):
    return {'name': name, 'irradianceCoefficients': [[1, 1, 1]] + [[0, 0, 0]] * 8,
            'specularImageSize': 4,
            'specularImages': [['probes/%s%d.png' % (tag, f) for f in range(6)]]}


def _zone(doc, name):
    node, = [node for node in doc['nodes'] if node.get('name') == name]
    return node['extensions']['OGLC_zone']


def test_a_baked_zone_names_its_light_and_no_longer_captures(tmp_path):
    path = _document(tmp_path)
    probes._rewrite(str(path), {'bore-0': _light('zone-0')})  # noqa: SLF001 white-box test of the helper
    doc = json.loads(path.read_text())
    first, second = _zone(doc, 'bore-0'), _zone(doc, 'bore-1')
    assert first['extensions']['EXT_lights_image_based'] == {'light': 0}
    assert 'capture' not in first['environment']
    assert first['environment']['intensity'] == 0.8
    assert 'capture' in second['environment']
    assert 'EXT_lights_image_based' in doc['extensionsUsed']
    light = doc['extensions']['EXT_lights_image_based']['lights'][0]
    assert [doc['images'][i]['uri'] for i in light['specularImages'][0]] == [
        'probes/a%d.png' % f for f in range(6)]


def test_a_zone_is_found_by_its_name_not_its_place_in_the_document(tmp_path):
    """The engine lists zones in the order it built them and leaves out any
    with a bad shape, so a position in that list is not a node's."""
    path = _document(tmp_path, names=('forest-1', 'bore-1'))
    probes._rewrite(str(path), {'bore-1': _light('bore')})  # noqa: SLF001 white-box test of the helper
    doc = json.loads(path.read_text())
    assert _zone(doc, 'bore-1')['extensions']['EXT_lights_image_based'] == {'light': 0}
    assert 'extensions' not in _zone(doc, 'forest-1')


def test_a_second_bake_replaces_a_zones_light_rather_than_adding_one(tmp_path):
    path = _document(tmp_path)
    probes._rewrite(str(path), {'bore-0': _light('zone-0', 'a'),  # noqa: SLF001 white-box test of the helper
                                'bore-1': _light('zone-1', 'b')})
    probes._rewrite(str(path), {'bore-0': _light('zone-0', 'c')})  # noqa: SLF001 white-box test of the helper
    doc = json.loads(path.read_text())
    lights = doc['extensions']['EXT_lights_image_based']['lights']
    assert len(lights) == 2
    assert len(doc['images']) == 12
    uris = {doc['images'][i]['uri'] for light in lights
            for level in light['specularImages'] for i in level}
    assert uris == {'probes/%s%d.png' % (tag, f) for tag in 'bc' for f in range(6)}
    first = _zone(doc, 'bore-0')['extensions']['EXT_lights_image_based']['light']
    second = _zone(doc, 'bore-1')['extensions']['EXT_lights_image_based']['light']
    assert lights[first]['specularImages'][0][0] == doc['images'].index(
        {'uri': 'probes/c0.png', 'mimeType': 'image/png'})
    assert doc['images'][lights[second]['specularImages'][0][0]]['uri'] == 'probes/b0.png'


def test_the_faces_are_rgbd():
    data = probes._png(np.full((4, 4, 3), 6.0, 'f4'))  # noqa: SLF001 white-box test of the helper
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    assert pixels.shape == (4, 4, 4)
    assert pixels[0, 0, 0] / pixels[0, 0, 3] == 6.0


def test_a_world_without_zones_is_left_alone(tmp_path):
    assert probes.bake_probes(str(tmp_path)) == 0


def test_the_bake_leaves_the_environment_as_it_was(tmp_path, monkeypatch):
    """Every later context in the process reads these."""
    for name in ('OPENGLCONTEXT_RENDERER', 'OPENGLCONTEXT_PROFILE', 'OPENGLCONTEXT_IBL'):
        monkeypatch.delenv(name, raising=False)
    _document(tmp_path)
    probes.bake_probes(str(tmp_path))
    assert not {'OPENGLCONTEXT_RENDERER', 'OPENGLCONTEXT_PROFILE',
                'OPENGLCONTEXT_IBL'} & set(os.environ)
