"""Baking zone environments into a world: the document the bake leaves behind."""
import json

import numpy as np

from OpenGLContext_editor.bake import probes
from OpenGLContext_editor.bake.zones import ZoneRecord, ZonesLayer


def _document(tmp_path):
    zones = [ZoneRecord(name='bore-%d' % i, centre=(i * 10.0, 0, 0),
                        rotation=(0, 0, 0, 1), size=(8, 8, 8),
                        environment={'capture': {'position': [0, 1, 0]}, 'intensity': 0.8})
             for i in range(3)]
    path = tmp_path / 'zones.gltf'
    path.write_text(json.dumps(ZonesLayer(zones, {}).document()))
    return path


def test_a_baked_zone_names_its_light_and_no_longer_captures(tmp_path):
    path = _document(tmp_path)
    lights = [{'name': 'zone-0', 'irradianceCoefficients': [[1, 1, 1]] + [[0, 0, 0]] * 8,
               'specularImageSize': 4,
               'specularImages': [['probes/a%d.png' % f for f in range(6)]]}]
    probes._rewrite(str(path), {0: 0}, lights)
    doc = json.loads(path.read_text())
    first, second = (node['extensions']['OGLC_zone'] for node in doc['nodes'][:2])
    assert first['extensions']['EXT_lights_image_based'] == {'light': 0}
    assert 'capture' not in first['environment']
    assert first['environment']['intensity'] == 0.8
    assert 'capture' in second['environment']
    assert 'EXT_lights_image_based' in doc['extensionsUsed']
    light = doc['extensions']['EXT_lights_image_based']['lights'][0]
    assert [doc['images'][i]['uri'] for i in light['specularImages'][0]] == [
        'probes/a%d.png' % f for f in range(6)]


def test_the_faces_are_rgbd(tmp_path):
    import io

    from PIL import Image
    data = probes._png(np.full((4, 4, 3), 6.0, 'f4'))
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    assert pixels.shape == (4, 4, 4)
    assert pixels[0, 0, 0] / pixels[0, 0, 3] == 6.0


def test_a_world_without_zones_is_left_alone(tmp_path):
    assert probes.bake_probes(str(tmp_path)) == 0
