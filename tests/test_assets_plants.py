"""Turning a scanned plant into ground cover the engine can grow (no network).

A published plant model is authored for a render: node transforms not yet
applied, a mesh per clump, tens of thousands of triangles, and -- because the
base colour ships as a JPEG -- the cutout mask in a file of its own. A field
wants the opposite of all of that: one mesh per plant, a few hundred triangles,
and one RGBA texture with the mask already in its alpha.

These cover the bake that crosses that gap. Everything here is CPU and local;
the fetch and the impostor card have their own.
"""
import json
import struct

import numpy as np
import pytest
from OpenGLContext.scenegraph.vegetation.clumps import load_clump_glb
from PIL import Image

from OpenGLContext_editor.assets import plants

CT_FLOAT, CT_UINT = 5126, 5125


def _gltf(directory, nodes, scale=1.0):
    """A model shaped the way a published scanned plant is.

    A separate ``.gltf``, its buffer beside it, its images beside that, one
    material over every mesh, and each clump in a node of its own with the
    transform that places it still to be applied.
    """
    blobs, views, accessors, meshes, node_json = bytearray(), [], [], [], []

    def view(data):
        while len(blobs) % 4:
            blobs.append(0)
        views.append({'buffer': 0, 'byteOffset': len(blobs),
                      'byteLength': len(data)})
        blobs.extend(data)
        return len(views) - 1

    for name, offset, *rest in nodes:
        stretch = rest[0] if rest else (1.0, 1.0, 1.0)
        # An upright card: 1 unit wide, `scale` tall, standing on y=0.
        P = np.array([(-0.5, 0, 0), (0.5, 0, 0), (0.5, scale, 0),
                      (-0.5, scale, 0)], '<f4')
        N = np.tile(np.array([(0, 0, 1)], '<f4'), (4, 1))
        UV = np.array([(0, 1), (1, 1), (1, 0), (0, 0)], '<f4')
        idx = np.array([0, 1, 2, 0, 2, 3], '<u4')
        first = len(accessors)
        accessors.append({'bufferView': view(P.tobytes()),
                          'componentType': CT_FLOAT, 'count': 4, 'type': 'VEC3',
                          'min': P.min(0).tolist(), 'max': P.max(0).tolist()})
        accessors.append({'bufferView': view(N.tobytes()),
                          'componentType': CT_FLOAT, 'count': 4, 'type': 'VEC3'})
        accessors.append({'bufferView': view(UV.tobytes()),
                          'componentType': CT_FLOAT, 'count': 4, 'type': 'VEC2'})
        accessors.append({'bufferView': view(idx.tobytes()),
                          'componentType': CT_UINT, 'count': 6, 'type': 'SCALAR'})
        meshes.append({'name': name, 'primitives': [{
            'attributes': {'POSITION': first, 'NORMAL': first + 1,
                           'TEXCOORD_0': first + 2},
            'indices': first + 3, 'material': 0}]})
        node_json.append({'name': name, 'mesh': len(meshes) - 1,
                          'translation': list(offset), 'scale': list(stretch)})

    (directory / 'plant.bin').write_bytes(bytes(blobs))
    Image.new('RGB', (8, 8), (30, 90, 40)).save(directory / 'diff.jpg')
    mask = np.zeros((8, 8), 'u1')
    mask[2:6, 2:6] = 255                       # a square of solid in a clear field
    Image.fromarray(mask, 'L').save(directory / 'alpha.png')

    document = {
        'asset': {'version': '2.0'},
        'scene': 0, 'scenes': [{'nodes': list(range(len(node_json)))}],
        'nodes': node_json, 'meshes': meshes,
        'buffers': [{'uri': 'plant.bin', 'byteLength': len(blobs)}],
        'bufferViews': views, 'accessors': accessors,
        'materials': [{'name': 'plant', 'alphaMode': 'BLEND',
                       'doubleSided': True,
                       'pbrMetallicRoughness': {
                           'baseColorTexture': {'index': 0}}}],
        'textures': [{'source': 0}],
        'images': [{'uri': 'diff.jpg', 'mimeType': 'image/jpeg'}],
    }
    path = directory / 'plant.gltf'
    path.write_text(json.dumps(document))
    return str(path)


def _source(tmp_path, nodes=(('fern_a', (0, 0, 0)), ('fern_b', (3, 0, 0))),
            scale=1.0):
    path = _gltf(tmp_path, list(nodes), scale=scale)
    return plants.PlantSource(
        slug='fern_02', gltf=path, diffuse=str(tmp_path / 'diff.jpg'),
        mask=str(tmp_path / 'alpha.png'),
        credit='Fern 02 by Somebody (Poly Haven, CC0)')


class TestReadingWhatWasPublished:
    def test_every_clump_in_the_file_becomes_a_variant(self, tmp_path) -> None:
        """One file is often several plants; each is a kind of its own."""
        assert [one.name for one in plants.flatten(_source(tmp_path).gltf)] \
            == ['fern_a', 'fern_b']

    def test_a_node_transform_is_applied(self, tmp_path) -> None:
        """The size a plant is drawn at is in its node, not in its mesh."""
        source = _source(tmp_path, nodes=[('small', (0, 0, 0)),
                                          ('tall', (3, 0, 0), (1, 3, 1))])
        small, tall = plants.flatten(source.gltf)
        assert tall.height == pytest.approx(3.0 * small.height)

    def test_a_plant_is_stood_at_the_origin(self, tmp_path) -> None:
        """Instanced at a position, so the model's own is in the way."""
        for one in plants.flatten(_source(tmp_path).gltf):
            assert abs(float(one.positions[:, 0].mean())) < 1e-4
            assert abs(float(one.positions[:, 2].mean())) < 1e-4
            assert float(one.positions[:, 1].min()) == pytest.approx(0.0)

    def test_its_real_height_is_kept(self, tmp_path) -> None:
        """What it is in metres, which is what a species is sized by."""
        found = plants.flatten(_source(tmp_path, scale=0.4).gltf)
        assert found[0].height == pytest.approx(0.4)

    def test_a_file_with_no_geometry_says_so(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            plants.flatten(_gltf(tmp_path, []))

    def test_the_publishers_level_of_detail_suffix_is_not_part_of_the_name(
            self, tmp_path) -> None:
        """Published nodes are named `fern_tall_a_LOD0`. The rungs here are
        this bake's own, so carrying someone else's numbering into a species
        name means two different things called LOD in one asset."""
        source = _source(tmp_path, nodes=[('fern_tall_a_LOD0', (0, 0, 0))])
        assert plants.flatten(source.gltf)[0].name == 'fern_tall_a'


class TestKeepingOnlySomeOfWhatIsInTheFile:
    """One published grass is seventeen tufts, and a field that scatters
    seventeen species pays for seventeen scatters and sixty-eight draws to show
    what a handful of them already shows."""

    def _many(self, tmp_path):
        """Four plants of the same height and very different substance."""
        source = _source(tmp_path, nodes=[('base', (0, 0, 0))])
        one = plants.flatten(source.gltf)[0]
        return source, [
            _subdivided(plants.replace_name(one, 'wisp'), 60),
            _subdivided(plants.replace_name(one, 'sparse'), 200),
            _subdivided(plants.replace_name(one, 'full'), 3000),
            _subdivided(plants.replace_name(one, 'dense'), 6000)]

    def test_all_of_them_unless_told_otherwise(self, tmp_path) -> None:
        source, found = self._many(tmp_path)
        assert len(plants.bake(source, str(tmp_path / 'out'), card=False,
                               variants=found)) == 4

    def test_it_keeps_the_ones_with_a_plant_actually_in_them(self,
                                                             tmp_path) -> None:
        """Substance, not height: the tallest tufts of a published grass are
        its leggy seed stalks, which is the one thing a card cannot show."""
        source, found = self._many(tmp_path)
        kept = plants.bake(source, str(tmp_path / 'out'), card=False,
                           variants=found, keep=2)
        assert sorted(one.name for one in kept) == ['dense', 'full']

    def test_asking_for_more_than_there_are_is_all_of_them(self,
                                                           tmp_path) -> None:
        source, found = self._many(tmp_path)
        assert len(plants.bake(source, str(tmp_path / 'out'), card=False,
                               variants=found, keep=99)) == 4


class TestTheMaskTheModelCameWithout:
    def test_the_cutout_carries_the_mask_as_its_alpha(self, tmp_path) -> None:
        """The published base colour is a JPEG, so it cannot hold one."""
        source = _source(tmp_path)
        texture = plants.cutout(source.diffuse, source.mask)
        assert texture.mode == 'RGBA'
        alpha = np.asarray(texture)[..., 3]
        assert alpha.max() == 255 and alpha.min() == 0

    def test_the_colour_is_the_one_published(self, tmp_path) -> None:
        source = _source(tmp_path)
        texture = np.asarray(plants.cutout(source.diffuse, source.mask))
        # Within what a JPEG round trip costs: the published colour is lossy
        # and is carried across as it was published, not re-authored.
        assert np.allclose(texture[0, 0, :3], (30, 90, 40), atol=3)

    def test_a_mask_of_another_size_is_made_to_fit(self, tmp_path) -> None:
        _source(tmp_path)
        Image.new('L', (4, 4), 200).save(tmp_path / 'small.png')
        texture = plants.cutout(str(tmp_path / 'diff.jpg'),
                                str(tmp_path / 'small.png'))
        assert texture.size == (8, 8)

    def test_solid_geometry_needs_no_mask(self, tmp_path) -> None:
        """A trunk is not a cutout; it is opaque all over."""
        _source(tmp_path)
        texture = plants.cutout(str(tmp_path / 'diff.jpg'), None)
        assert (np.asarray(texture)[..., 3] == 255).all()


class TestWhatTheBakeWrites:
    def _baked(self, tmp_path, **named):
        source = _source(tmp_path)
        out = tmp_path / 'out'
        out.mkdir()
        return source, out, plants.bake(source, str(out), card=False, **named)

    def test_a_species_comes_back_for_every_variant(self, tmp_path) -> None:
        _source_, _out, species = self._baked(tmp_path)
        assert [one.name for one in species] == ['fern_a', 'fern_b']

    def test_every_species_points_at_the_one_file(self, tmp_path) -> None:
        """A megabyte of texture is not carried once per rung."""
        _source_, out, species = self._baked(tmp_path)
        assert {one.clump for one in species} == {'fern_02.glb'}
        assert (out / 'fern_02.glb').exists()

    def test_the_texture_is_embedded_once_for_all_of_them(self, tmp_path) -> None:
        """Four variants at two rungs would otherwise carry it eight times."""
        _source_, out, species = self._baked(tmp_path)
        with open(out / species[0].clump, 'rb') as handle:
            handle.seek(12)
            length, _kind = struct.unpack('<II', handle.read(8))
            document = json.loads(handle.read(length))
        assert len(document['meshes']) == 4        # two variants, two rungs each
        assert len(document['images']) == 1

    def test_each_variant_names_its_own_two_rungs(self, tmp_path) -> None:
        _source_, _out, species = self._baked(tmp_path)
        assert species[0].clump_mesh == 'fern_a'
        assert species[0].clump_far_mesh == 'fern_a_far'

    def test_what_it_wrote_is_what_the_engine_reads(self, tmp_path) -> None:
        _source_, out, species = self._baked(tmp_path)
        path = str(out / species[0].clump)
        points, _n, _uv, indices, texture = load_clump_glb(
            path, mesh=species[0].clump_mesh)
        assert len(indices) and len(points)
        assert texture.mode == 'RGBA'
        assert float(points[:, 1].max()) == pytest.approx(1.0)   # height-normalised

    def test_a_species_is_as_tall_as_the_plant_was(self, tmp_path) -> None:
        source = _source(tmp_path, scale=0.35)
        out = tmp_path / 'out'
        out.mkdir()
        species = plants.bake(source, str(out), card=False)
        assert species[0].height == pytest.approx(0.35)

    def test_the_far_rung_is_no_dearer_than_the_near_one(self, tmp_path) -> None:
        _source_, out, species = self._baked(tmp_path)
        path = str(out / species[0].clump)
        near = load_clump_glb(path, mesh=species[0].clump_mesh)[3]
        far = load_clump_glb(path, mesh=species[0].clump_far_mesh)[3]
        assert len(far) <= len(near)

    def test_a_dense_plant_is_brought_down_to_the_budget(self, tmp_path) -> None:
        """A shrub is authored at tens of thousands of triangles; a field of
        them is not."""
        source = _source(tmp_path, nodes=[('bush', (0, 0, 0))])
        out = tmp_path / 'out'
        out.mkdir()
        dense = plants.flatten(source.gltf)[0]
        grid = _subdivided(dense, 4000)
        species = plants.bake(source, str(out), card=False,
                              variants=[grid], rungs=(300, 80))
        path = str(out / species[0].clump)
        assert len(load_clump_glb(path, mesh='bush')[3]) // 3 <= 320
        assert len(load_clump_glb(path, mesh='bush_far')[3]) // 3 <= 90

    def test_it_writes_what_the_plant_has_to_be_credited_as(self, tmp_path) -> None:
        source, out, _species = self._baked(tmp_path)
        assert source.credit in (out / 'CREDITS.txt').read_text()


def _subdivided(variant, triangles):
    """The same card, cut into roughly ``triangles`` triangles.

    A stand-in for a scanned plant's tessellation, so the bake has something to
    actually reduce.
    """
    side = max(int(np.sqrt(triangles / 2)), 2)
    u, v = np.meshgrid(np.linspace(0, 1, side + 1), np.linspace(0, 1, side + 1))
    positions = np.stack([u.ravel() - 0.5, v.ravel(), np.zeros(u.size)],
                         1).astype('f4')
    normals = np.tile(np.array([(0, 0, 1)], 'f4'), (len(positions), 1))
    uvs = np.stack([u.ravel(), v.ravel()], 1).astype('f4')
    quad = np.arange(side * side).reshape(side, side)
    base = (quad // side) * (side + 1) + (quad % side)
    corners = np.stack([base, base + 1, base + side + 2, base + side + 1],
                       -1).reshape(-1, 4)
    indices = np.concatenate([corners[:, [0, 1, 2]],
                              corners[:, [0, 2, 3]]]).ravel().astype('u4')
    return plants.Variant(name=variant.name, positions=positions,
                          normals=normals, uvs=uvs, indices=indices,
                          height=variant.height)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
