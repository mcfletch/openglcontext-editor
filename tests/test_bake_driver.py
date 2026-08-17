"""The bake itself: layers in, a streamable tileset on disk out.

These drive the whole spine -- partition, per-node content, glTF write, tileset
write -- and then read the result back the way the engine does: parse the
tileset with the runtime's own parser, and load each tile's content with the
glTF loader. The invariant that matters most is the last one asserted here: a
tile's geometry lies inside the bounding volume the tileset claims for it, or
the traversal culls content that is still on screen.
"""

import json
import os

import numpy as np
import pytest
from OpenGLContext.loaders import gltf
from OpenGLContext.loaders.tiles3d.tileset import build_runtime_tileset
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.driver import BakeResult, bake_world
from OpenGLContext_editor.bake.layers import HeightfieldLayer, InstanceLayer

EXTENT = BoundingBox((-256, 0, -256), (256, 0, 256))


def _hills(x, z):
    x, z = np.asarray(x, 'd'), np.asarray(z, 'd')
    return 20.0 * np.sin(x / 90.0) + 12.0 * np.cos(z / 70.0)


def _terrain(**kwargs):
    kwargs.setdefault('resolution', 9)
    kwargs.setdefault('extent', EXTENT)
    return HeightfieldLayer(height_fn=_hills, **kwargs)


def _trees(count=120):
    rng = np.random.default_rng(11)
    xz = rng.uniform(-250, 250, size=(count, 2))
    positions = np.stack([xz[:, 0], _hills(xz[:, 0], xz[:, 1]), xz[:, 1]], axis=-1)
    trunk = PBRMesh(positions=np.array([(0, 0, 0), (1, 0, 0), (0, 6, 0)], 'f'),
                    indices=np.array([0, 1, 2], np.uint32))
    return InstanceLayer(positions=positions, lods=[(0.0, trunk)], name='trees')


def _bake(tmp_path, layers=None, **kwargs):
    kwargs.setdefault('depth', 2)
    return bake_world(layers or [_terrain()], str(tmp_path), **kwargs)


def _parse(result):
    with open(result.tileset) as handle:
        document = json.load(handle)
    return build_runtime_tileset(document, base_uri=result.directory + os.sep,
                                 recenter=True)


class TestTheBakeProduces:
    def test_a_tileset_on_disk(self, tmp_path) -> None:
        result = _bake(tmp_path)
        assert os.path.exists(result.tileset)
        assert isinstance(result, BakeResult)

    def test_a_tile_per_node_of_the_tree(self, tmp_path) -> None:
        """Depth 2 over a surface world: 1 + 4 + 16 nodes, all with ground."""
        result = _bake(tmp_path, depth=2)
        assert result.tiles == 21

    def test_content_files_the_tileset_names(self, tmp_path) -> None:
        result = _bake(tmp_path)
        for tile in _parse(result).root.iter_tiles():
            for uri in tile.content_uris:
                assert os.path.exists(uri), uri

    def test_a_report_of_what_it_wrote(self, tmp_path) -> None:
        result = _bake(tmp_path, layers=[_terrain(), _trees()])
        assert result.contents == result.tiles           # every tile has content
        assert result.bytes_written > 0
        assert result.layers['terrain'] == 21
        assert result.layers['trees'] > 0

    def test_progress_is_reported_as_it_goes(self, tmp_path) -> None:
        seen = []
        _bake(tmp_path, progress=lambda done, total: seen.append((done, total)))
        assert seen and seen[-1][0] == seen[-1][1]

    def test_credits_ride_with_the_world(self, tmp_path) -> None:
        _bake(tmp_path, credits=['Elevation: SRTM, public domain'])
        assert 'SRTM' in (tmp_path / 'CREDITS.txt').read_text()


class TestTheTilesetItProduces:
    def test_the_runtime_parses_it(self, tmp_path) -> None:
        parsed = _parse(_bake(tmp_path))
        assert len(list(parsed.root.iter_tiles())) == 21

    def test_the_error_falls_as_the_tree_descends(self, tmp_path) -> None:
        parsed = _parse(_bake(tmp_path, depth=3))
        for tile in parsed.root.iter_tiles():
            for child in tile.children:
                assert child.geometric_error <= tile.geometric_error

    def test_the_leaves_claim_no_error(self, tmp_path) -> None:
        """Nothing finer exists, so a leaf must not ask to be refined."""
        parsed = _parse(_bake(tmp_path))
        for tile in parsed.root.iter_tiles():
            if not tile.children:
                assert tile.geometric_error == 0.0

    def test_it_refines_by_replacement(self, tmp_path) -> None:
        assert _parse(_bake(tmp_path)).root.refine == 'REPLACE'

    def test_the_root_error_can_be_set(self, tmp_path) -> None:
        parsed = _parse(_bake(tmp_path, root_error=64.0))
        assert parsed.root.geometric_error == 64.0

    def test_a_region_with_nothing_in_it_gets_no_tile(self, tmp_path) -> None:
        """Half the world empty means half the tree is never written."""
        half = BoundingBox((-256, 0, -256), (0, 0, 256))
        result = _bake(tmp_path, layers=[_terrain(extent=half)], depth=2,
                       bounds=EXTENT.with_height(-64, 64))
        assert result.tiles == 1 + 2 + 8


class TestTheContentItProduces:
    def test_every_tile_loads_through_the_gltf_loader(self, tmp_path) -> None:
        for tile in _parse(_bake(tmp_path)).root.iter_tiles():
            for uri in tile.content_uris:
                scene = gltf.load_gltf(uri)
                assert scene.group is not None

    def test_geometry_lies_inside_the_bounding_volume_claimed_for_it(
            self, tmp_path) -> None:
        result = _bake(tmp_path, layers=[_terrain(), _trees()], depth=2)
        with open(result.tileset) as handle:
            document = json.load(handle)
        for _depth, entry in _walk(document['root']):
            box = _box_of(entry)
            for uri in _uris(entry):
                points = _points(os.path.join(result.directory, uri))
                assert np.all(points.min(axis=0) >= box.minimum - 1e-3), uri
                assert np.all(points.max(axis=0) <= box.maximum + 1e-3), uri

    def test_a_child_tile_is_finer_ground_than_its_parent(self, tmp_path) -> None:
        """Same vertex budget over a quarter of the ground."""
        result = _bake(tmp_path, depth=1)
        with open(result.tileset) as handle:
            document = json.load(handle)
        root, child = document['root'], document['root']['children'][0]
        assert _spacing(result, root) > _spacing(result, child)

    def test_instances_are_written_as_gpu_instancing(self, tmp_path) -> None:
        result = _bake(tmp_path, layers=[_trees()], depth=1)
        with open(result.tileset) as handle:
            document = json.load(handle)
        uri = _uris(document['root'])[0]
        from OpenGLContext.loaders.gltf.writer import _pack_glb  # noqa: F401
        raw = open(os.path.join(result.directory, uri), 'rb').read()
        assert b'EXT_mesh_gpu_instancing' in raw

    def test_every_instance_is_somewhere_in_the_tree(self, tmp_path) -> None:
        """Thinning a coarse tile must not lose a tree from the world."""
        trees = _trees(count=200)
        result = _bake(tmp_path, layers=[trees], depth=3, max_instances=16)
        placed = set()
        with open(result.tileset) as handle:
            document = json.load(handle)
        for _tile, entry in _walk(document['root']):
            for uri in _uris(entry):
                scene = gltf.load_gltf(os.path.join(result.directory, uri))
                for node in _flatten(scene.group):
                    translation = getattr(node, 'translation', None)
                    if translation is not None and getattr(node, 'children', None):
                        placed.add(tuple(round(float(v), 2) for v in translation))
        expected = {tuple(round(float(v), 2) for v in row) for row in trees.positions}
        assert placed >= expected


# --- helpers ------------------------------------------------------------------

def _walk(entry, depth=0):
    yield depth, entry
    for child in entry.get('children', []):
        yield from _walk(child, depth + 1)


def _uris(entry):
    if 'content' in entry:
        return [entry['content']['uri']]
    return [c['uri'] for c in entry.get('contents', [])]


def _box_of(entry):
    """The Y-up box a tile's Z-up bounding volume describes."""
    volume = entry['boundingVolume']['box']
    center = np.array([volume[0], volume[2], -volume[1]])
    half = np.array([abs(volume[3]), abs(volume[11]), abs(volume[7])])
    return BoundingBox(center - half, center + half)


def _points(path):
    """Every vertex of a tile, in world coordinates.

    Instanced content arrives as one shared mesh under a Transform per
    placement, so the walk carries the translation and scale down with it.
    """
    out = []
    _collect(gltf.load_gltf(path).group, np.zeros(3), np.ones(3), out)
    assert out, path
    return np.vstack(out)


def _collect(node, offset, scale, out):
    translation = getattr(node, 'translation', None)
    if translation is not None:
        offset = offset + np.asarray(translation, 'd') * scale
    node_scale = getattr(node, 'scale', None)
    if node_scale is not None:
        scale = scale * np.asarray(node_scale, 'd')
    geometry = getattr(node, 'geometry', None)
    if geometry is not None and getattr(geometry, 'positions', None) is not None:
        out.append(np.asarray(geometry.positions, 'd') * scale + offset)
    for child in getattr(node, 'children', None) or []:
        _collect(child, offset, scale, out)


def _flatten(node, out=None):
    out = [] if out is None else out
    out.append(node)
    for child in getattr(node, 'children', None) or []:
        _flatten(child, out)
    return out


def _spacing(result, entry):
    """Ground covered per vertex, the measure of how fine a terrain tile is."""
    points = _points(os.path.join(result.directory, _uris(entry)[0]))
    width = points[:, 0].max() - points[:, 0].min()
    return width / np.sqrt(len(points))


class TestWhatItRefuses:
    def test_a_bake_with_no_layers_says_so(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='nothing to bake'):
            bake_world([], str(tmp_path))

    def test_a_world_with_content_nowhere_says_so(self, tmp_path) -> None:
        empty = HeightfieldLayer(height_fn=_hills,
                                 extent=BoundingBox((-1, 0, -1), (1, 0, 1)))
        with pytest.raises(ValueError, match='nothing to bake'):
            bake_world([empty], str(tmp_path), depth=0,
                       bounds=BoundingBox((500, 500, 500), (600, 600, 600)))


class TestTheReportItPrints:
    def test_the_summary_names_what_was_written(self, tmp_path) -> None:
        result = _bake(tmp_path, layers=[_terrain(), _trees()])
        assert 'tiles' in result.summary() and 'MB' in result.summary()

    def test_the_long_report_lists_the_layers(self, tmp_path) -> None:
        from OpenGLContext_editor.bake.driver import bake_summary
        report = bake_summary(_bake(tmp_path, layers=[_terrain(), _trees()]))
        assert 'terrain' in report and 'trees' in report
        assert 'extent:' in report and 'tileset:' in report
