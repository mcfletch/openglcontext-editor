"""The storage forms a published plant may arrive in, read as the engine reads them.

A glTF may store an attribute interleaved with others, quantized to integers
the reader scales back (``KHR_mesh_quantization``), sparse, or not indexed at
all, and a primitive may be a strip rather than a list. A plant read wrongly is
a bake of garbage, so each form is read here and compared with the plain one.
The document is also data from the network: a buffer it names is read from
beside it and from nowhere else.
"""
import json

import numpy as np
import pytest

from OpenGLContext_editor.assets import plants

FLOAT, USHORT, UINT = 5126, 5123, 5125

#: A unit card standing on y=0, as a triangle list.
CARD = np.array([(-0.5, 0, 0), (0.5, 0, 0), (0.5, 1, 0), (-0.5, 1, 0)], '<f4')
CARD_UV = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], '<f4')
CARD_LIST = np.array([0, 1, 2, 0, 2, 3], '<u4')


class _Document:
    """A one-node, one-primitive glTF assembled piece by piece."""

    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []
        self.attributes: dict[str, int] = {}
        self.primitive: dict = {}

    def view(self, data: bytes, stride: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        entry = {'buffer': 0, 'byteOffset': len(self.blob),
                 'byteLength': len(data)}
        if stride:
            entry['byteStride'] = stride
        self.views.append(entry)
        self.blob.extend(data)
        return len(self.views) - 1

    def accessor(self, view: int | None, component: int, count: int,
                 kind: str, **extra) -> int:
        entry = {'componentType': component, 'count': count, 'type': kind,
                 **extra}
        if view is not None:
            entry['bufferView'] = view
        self.accessors.append(entry)
        return len(self.accessors) - 1

    def plain(self, points=CARD, uv=CARD_UV) -> None:
        self.attributes['POSITION'] = self.accessor(
            self.view(points.tobytes()), FLOAT, len(points), 'VEC3',
            min=points.min(0).tolist(), max=points.max(0).tolist())
        self.attributes['TEXCOORD_0'] = self.accessor(
            self.view(uv.tobytes()), FLOAT, len(uv), 'VEC2')

    def indexed(self, indices=CARD_LIST) -> None:
        self.primitive['indices'] = self.accessor(
            self.view(indices.tobytes()), UINT, len(indices), 'SCALAR')

    def write(self, directory, uri='plant.bin') -> str:
        (directory / 'plant.bin').write_bytes(bytes(self.blob))
        document = {
            'asset': {'version': '2.0'},
            'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'name': 'tuft', 'mesh': 0}],
            'meshes': [{'name': 'tuft', 'primitives': [
                dict(self.primitive, attributes=self.attributes)]}],
            'buffers': [{'uri': uri, 'byteLength': len(self.blob)}],
            'bufferViews': self.views, 'accessors': self.accessors,
        }
        path = directory / 'plant.gltf'
        path.write_text(json.dumps(document))
        return str(path)


def _triangles(variant) -> set:
    """The triangles of a variant as sets of corner positions, for comparing
    two readings of one surface however their vertices are numbered."""
    corners = np.round(variant.positions[variant.indices.reshape(-1, 3)], 5)
    return {frozenset(map(tuple, triangle)) for triangle in corners}


@pytest.fixture
def plain(tmp_path):
    (tmp_path / 'plain').mkdir()
    document = _Document()
    document.plain()
    document.indexed()
    return plants.flatten(document.write(tmp_path / 'plain'))[0]


class TestTheFormsAnAttributeMayTake:
    def test_interleaved_positions_and_coordinates(self, tmp_path,
                                                   plain) -> None:
        document = _Document()
        rows = np.zeros(4, [('p', '<f4', 3), ('uv', '<f4', 2)])
        rows['p'], rows['uv'] = CARD, CARD_UV
        shared = document.view(rows.tobytes(), stride=rows.itemsize)
        document.attributes['POSITION'] = document.accessor(
            shared, FLOAT, 4, 'VEC3', min=CARD.min(0).tolist(),
            max=CARD.max(0).tolist())
        document.attributes['TEXCOORD_0'] = document.accessor(
            shared, FLOAT, 4, 'VEC2', byteOffset=12)
        document.indexed()
        found = plants.flatten(document.write(tmp_path))[0]
        assert _triangles(found) == _triangles(plain)
        np.testing.assert_allclose(found.uvs, plain.uvs)

    def test_quantized_texture_coordinates_are_scaled_back(self, tmp_path,
                                                           plain) -> None:
        """KHR_mesh_quantization stores a coordinate as a normalized integer."""
        document = _Document()
        document.plain()
        quantized = np.round(CARD_UV * 65535).astype('<u2')
        document.attributes['TEXCOORD_0'] = document.accessor(
            document.view(quantized.tobytes()), USHORT, 4, 'VEC2',
            normalized=True)
        document.indexed()
        found = plants.flatten(document.write(tmp_path))[0]
        np.testing.assert_allclose(found.uvs, plain.uvs, atol=1e-4)

    def test_a_sparse_accessor_overrides_its_base(self, tmp_path) -> None:
        """A sparse accessor with no base is zeros with the listed rows set."""
        document = _Document()
        document.plain()
        rows = np.array([2, 3], '<u4')
        lifted = np.array([(0.5, 2, 0), (-0.5, 2, 0)], '<f4')
        base = document.attributes['POSITION']
        document.accessors[base]['sparse'] = {
            'count': 2,
            'indices': {'bufferView': document.view(rows.tobytes()),
                        'componentType': UINT},
            'values': {'bufferView': document.view(lifted.tobytes())}}
        document.accessors[base]['max'] = [0.5, 2.0, 0.0]
        document.indexed()
        found = plants.flatten(document.write(tmp_path))[0]
        assert found.height == pytest.approx(2.0)


class TestTheFormsAPrimitiveMayTake:
    def test_a_strip_is_read_as_the_triangles_it_draws(self, tmp_path,
                                                       plain) -> None:
        document = _Document()
        document.plain()
        document.indexed(np.array([1, 2, 0, 3], '<u4'))
        document.primitive['mode'] = 5                # TRIANGLE_STRIP
        found = plants.flatten(document.write(tmp_path))[0]
        assert _triangles(found) == _triangles(plain)

    def test_an_unindexed_primitive_is_not_dropped(self, tmp_path,
                                                   plain) -> None:
        document = _Document()
        document.plain(CARD[CARD_LIST], CARD_UV[CARD_LIST])
        found = plants.flatten(document.write(tmp_path))[0]
        assert _triangles(found) == _triangles(plain)

    def test_lines_are_not_a_plant(self, tmp_path) -> None:
        document = _Document()
        document.plain()
        document.indexed(np.array([0, 1, 1, 2], '<u4'))
        document.primitive['mode'] = 1                # LINES
        with pytest.raises(ValueError):
            plants.flatten(document.write(tmp_path))


class TestWhereABufferMayBeRead:
    """The document came off the network, and what it names is data."""

    @pytest.mark.parametrize('uri', ['../secret.bin', '/etc/hostname',
                                     'file:///etc/hostname'])
    def test_a_buffer_outside_the_download_is_refused(self, tmp_path,
                                                      uri) -> None:
        (tmp_path / 'secret.bin').write_bytes(b'\0' * 256)
        (tmp_path / 'download').mkdir()
        document = _Document()
        document.plain()
        document.indexed()
        with pytest.raises(OSError):
            plants.flatten(document.write(tmp_path / 'download', uri=uri))


class TestWhereAPlantIsStood:
    def test_on_the_middle_of_its_spread_not_of_its_vertices(self,
                                                             tmp_path) -> None:
        """A dense clump to one side pulls the vertex mean toward it; the
        instanced plant should stand on the middle of what it covers."""
        document = _Document()
        spread = [CARD + (x, 0, 0) for x in (0.0, 0.1, 0.2, 0.3, 4.0)]
        points = np.concatenate(spread).astype('<f4')
        indices = np.concatenate([CARD_LIST + 4 * i for i in range(5)])
        document.plain(points, np.tile(CARD_UV, (5, 1)))
        document.indexed(indices.astype('<u4'))
        found = plants.flatten(document.write(tmp_path))[0]
        low, high = found.positions.min(0), found.positions.max(0)
        assert low[0] == pytest.approx(-high[0])
        assert low[2] == pytest.approx(-high[2])
