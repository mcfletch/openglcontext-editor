"""A published plant model, baked into ground cover the engine can grow.

What is published is authored for a render. What a field wants is the opposite
of that, in four ways, and this is the four:

**One mesh per plant.** A published file is a node per clump with the transform
that places it still to be applied -- often several plants in one file, which is
several kinds of cover for one download. :func:`flatten` applies the transforms
and hands back one :class:`Variant` per clump, stood at the origin on ``y=0``
where an instanced draw will put it.

**The mask the model came without.** These plants are alpha-cut cards, and the
published base colour is a JPEG, which cannot hold an alpha channel; the mask
ships as a map of its own. :func:`cutout` puts it back where the cutout shader
looks for it.

**A triangle count a field can afford.** A shrub authored at twenty-seven
thousand triangles is one shrub. :func:`bake` reduces each plant to two rungs --
what is drawn close up and what is drawn over the rest of the disc -- with
``opengl_decimate``, whose border handling is what keeps the silhouette of an
alpha card. One reduction serves both rungs, since each is a prefix of the same
recorded sequence of contractions.

**One texture, not one per rung.** A 1k RGBA cutout is about a megabyte, and a
plant of four variants at two rungs each would carry it eight times. So a bake
writes one ``.glb`` holding every rung of every variant against the single
embedded image they share, and each
:class:`~OpenGLContext.scenegraph.vegetation.cover.CoverSpecies` names the mesh
in it that is its own.

Normals are *carried* through the reduction rather than recomputed, so a
flattened triangle still shades the way the surface it replaced did -- the
finding recorded in :mod:`OpenGLContext_editor.meshlod.chain`, which builds a
whole chain from the same reduction.
"""
from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.vegetation.cover import CoverSpecies

__all__ = ['PlantSource', 'Variant', 'bake', 'cutout', 'flatten',
           'replace_name', 'NEAR_TRIANGLES', 'FAR_TRIANGLES']

#: What one plant is allowed to cost, close up and over the rest of the disc.
#: The outer ring of a cover disc is most of the plants in it, so the far rung
#: is where the triangles of a field actually are.
NEAR_TRIANGLES = 600
FAR_TRIANGLES = 150

#: How transparent a texel has to be before the cutout drops it. The same figure
#: the clump shader tests at, so what is baked is what is drawn.
ALPHA_CUTOFF = 0.33

#: A published node often carries the publisher's own level-of-detail number.
#: The rungs here are this bake's own, so carrying theirs into a species name
#: would leave two different things called LOD in the one asset.
_PUBLISHED_LOD = re.compile(r'_LOD\d+$', re.IGNORECASE)

#: glTF component types, as numpy reads them, and how many of each a type holds.
_COMPONENTS = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4',
               5126: '<f4'}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


@dataclass(frozen=True)
class PlantSource:
    """A downloaded model and the maps a cutout plant needs from beside it.

    ``mask`` is the alpha/opacity map published separately, or ``None`` for
    geometry that is solid all over and needs no cutout. ``credit`` is what the
    bake writes out beside the assets: CC0 asks for no attribution, and it is
    given anyway.
    """

    slug: str
    gltf: str
    diffuse: str
    mask: str | None = None
    credit: str = ''

    def __repr__(self) -> str:
        return 'PlantSource(%s)' % (self.slug,)


@dataclass(frozen=True)
class Variant:
    """One plant out of a model file, stood at the origin.

    ``height`` is what it measures in metres, which is what a species is sized
    by once the mesh itself has been normalised to a unit tall.
    """

    name: str
    positions: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray
    height: float

    @property
    def triangle_count(self) -> int:
        """Triangles in this plant as it was published."""
        return len(self.indices) // 3


def _document(path: str) -> tuple[dict, list[bytes]]:
    """A glTF document and its buffers, from a ``.gltf`` or a ``.glb``."""
    with open(path, 'rb') as handle:
        raw = handle.read()
    if raw[:4] == b'glTF':
        _magic, _version, length = struct.unpack('<III', raw[:12])
        offset, chunks = 12, []
        while offset < length:
            size, kind = struct.unpack('<II', raw[offset:offset + 8])
            chunks.append((kind, raw[offset + 8:offset + 8 + size]))
            offset += 8 + size
        document = json.loads(chunks[0][1])
        return document, [chunks[1][1] if len(chunks) > 1 else b'']
    document = json.loads(raw)
    beside = os.path.dirname(os.path.abspath(path))
    buffers = []
    for entry in document.get('buffers', ()):
        uri = entry.get('uri')
        if uri is None:
            raise ValueError("%s names a buffer with no uri and is not a .glb"
                             % (path,))
        if uri.startswith('data:'):
            import base64
            buffers.append(base64.b64decode(uri.split(',', 1)[1]))
        else:
            from urllib.parse import unquote
            with open(os.path.join(beside, unquote(uri)), 'rb') as handle:
                buffers.append(handle.read())
    return document, buffers


def _accessor(document: dict, buffers: list[bytes], index: int) -> np.ndarray:
    """One accessor as an ``(count, width)`` array."""
    entry = document['accessors'][index]
    view = document['bufferViews'][entry['bufferView']]
    width = _WIDTHS[entry['type']]
    start = view.get('byteOffset', 0) + entry.get('byteOffset', 0)
    stride = view.get('byteStride')
    data = buffers[view.get('buffer', 0)]
    if stride and stride != width * np.dtype(
            _COMPONENTS[entry['componentType']]).itemsize:
        # Interleaved: take the accessor's own slice out of each stride.
        item = np.dtype(_COMPONENTS[entry['componentType']]).itemsize * width
        rows = np.frombuffer(data, 'u1',
                             stride * (entry['count'] - 1) + item, start)
        picked = np.lib.stride_tricks.as_strided(
            rows, (entry['count'], item), (stride, 1)).copy()
        flat = np.frombuffer(picked.tobytes(),
                             _COMPONENTS[entry['componentType']])
    else:
        flat = np.frombuffer(data, _COMPONENTS[entry['componentType']],
                             entry['count'] * width, start)
    return flat.reshape(entry['count'], width)


def _local_matrix(node: dict) -> np.ndarray:
    """A node's own transform, as a 4x4."""
    if 'matrix' in node:
        return np.asarray(node['matrix'], 'd').reshape(4, 4).T
    matrix = np.eye(4)
    if 'scale' in node:
        matrix = matrix @ np.diag(list(node['scale']) + [1.0])
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        turn = np.eye(4)
        turn[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        matrix = turn @ matrix
    if 'translation' in node:
        move = np.eye(4)
        move[:3, 3] = node['translation']
        matrix = move @ matrix
    return matrix


def flatten(path: str) -> list[Variant]:
    """Every plant in a model file, transforms applied and stood at the origin.

    One :class:`Variant` per node that has geometry, its primitives merged --
    they share the one material, being one plant. Each is centred in plan and
    sat on ``y=0``, because an instanced draw supplies the position and the
    model's own is in the way of it.
    """
    document, buffers = _document(path)
    nodes = document.get('nodes', [])
    scene = document.get('scenes', [{}])[document.get('scene', 0)]
    found: list[Variant] = []

    def walk(index: int, parent: np.ndarray) -> None:
        node = nodes[index]
        matrix = parent @ _local_matrix(node)
        if 'mesh' in node:
            named = (node.get('name')
                     or document['meshes'][node['mesh']].get('name')
                     or 'part%d' % (index,))
            variant = _variant(document, buffers,
                               document['meshes'][node['mesh']], matrix,
                               _PUBLISHED_LOD.sub('', named))
            if variant is not None:
                found.append(variant)
        for child in node.get('children', ()):
            walk(child, matrix)

    for root in scene.get('nodes', range(len(nodes))):
        walk(root, np.eye(4))
    if not found:
        raise ValueError("%s holds no geometry to bake a plant from" % (path,))
    return found


def _variant(document: dict, buffers: list[bytes], mesh: dict,
             matrix: np.ndarray, name: str) -> Variant | None:
    """One node's primitives, merged and placed."""
    points, normals, uvs, indices, base = [], [], [], [], 0
    rotate = matrix[:3, :3]
    # A normal is transformed by the inverse transpose, or a non-uniform scale
    # leaves it off the surface it belongs to.
    straighten = np.linalg.inv(rotate).T
    for primitive in mesh.get('primitives', ()):
        attributes = primitive.get('attributes', {})
        if 'POSITION' not in attributes or 'indices' not in primitive:
            continue
        P = _accessor(document, buffers, attributes['POSITION']).astype('d')
        points.append((rotate @ P.T).T + matrix[:3, 3])
        if 'NORMAL' in attributes:
            N = _accessor(document, buffers, attributes['NORMAL']).astype('d')
            N = (straighten @ N.T).T
        else:
            N = np.tile(np.array([(0.0, 1.0, 0.0)]), (len(P), 1))
        lengths = np.linalg.norm(N, axis=1, keepdims=True)
        normals.append(N / np.where(lengths > 1e-12, lengths, 1.0))
        uvs.append(_accessor(document, buffers, attributes['TEXCOORD_0']
                             ).astype('f4')
                   if 'TEXCOORD_0' in attributes else np.zeros((len(P), 2), 'f4'))
        indices.append(_accessor(document, buffers,
                                 primitive['indices']).ravel() + base)
        base += len(P)
    if not points:
        return None
    P = np.concatenate(points)
    height = float(P[:, 1].max() - P[:, 1].min())
    P = P - [P[:, 0].mean(), P[:, 1].min(), P[:, 2].mean()]
    return Variant(name=name, positions=P.astype('f4'),
                   normals=np.concatenate(normals).astype('f4'),
                   uvs=np.concatenate(uvs).astype('f4'),
                   indices=np.concatenate(indices).astype('u4'),
                   height=height)


def replace_name(variant: Variant, name: str) -> Variant:
    """``variant`` under another name, for a caller assembling its own set."""
    from dataclasses import replace
    return replace(variant, name=name)


def cutout(diffuse: str, mask: str | None) -> Any:
    """The published base colour with its separately published mask as alpha.

    With no ``mask`` the result is opaque, which is what solid geometry wants.
    A mask published at another size is resampled to the colour's, so the two
    always agree about where a texel is.
    """
    from PIL import Image
    colour = Image.open(diffuse).convert('RGB')
    if mask is None:
        alpha = Image.new('L', colour.size, 255)
    else:
        alpha = Image.open(mask).convert('L')
        if alpha.size != colour.size:
            alpha = alpha.resize(colour.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*colour.split(), alpha))


def _rungs(variant: Variant,
           targets: Sequence[int]) -> list[tuple[dict, np.ndarray]]:
    """``variant``'s indices reduced to each of ``targets`` triangles.

    One reduction, read off at several counts: ``opengl_decimate`` records the
    contractions in order and every count is a prefix of that record, so the
    second rung costs nothing the first has not already paid for.
    """
    from opengl_decimate import SimplifyOptions, collapse_sequence
    attributes = {'POSITION': variant.positions, 'NORMAL': variant.normals,
                  'TEXCOORD_0': variant.uvs}
    smallest = min(targets)
    if variant.triangle_count <= smallest:
        return [(attributes, variant.indices) for _ in targets]
    sequence = collapse_sequence(
        attributes, variant.indices,
        SimplifyOptions(target_count=smallest, recompute_normals=False,
                        weld_tolerance=1e-5))
    out = []
    for target in targets:
        if variant.triangle_count <= target:
            out.append((attributes, variant.indices))
            continue
        result = sequence.at(target_count=int(target))
        out.append((result.attributes, result.indices))
    return out


def bake(source: PlantSource, directory: str,
         rungs: Sequence[int] = (NEAR_TRIANGLES, FAR_TRIANGLES),
         variants: Sequence[Variant] | None = None,
         keep: int | None = None,
         card: bool = True, card_size: int = 512,
         density: float = 0.35, patchiness: float = 0.0,
         patch_metres: float | None = None,
         canopy: tuple[float, float] | None = None) -> list[CoverSpecies]:
    """Bake ``source`` into ``directory`` and say what species it grew.

    Writes one ``<slug>.glb`` holding every variant at every rung against the
    one cutout texture, the billboard each variant becomes past its geometry
    (with ``card``, which needs a GL context), and the credit for the work.

    ``rungs`` is the triangle budget for the near and far geometry.
    ``density`` is plants per square metre, which the caller usually knows
    better than this does -- a grass is dense and a shrub is not. So are
    ``patchiness`` (how much the plant gathers into beds), ``patch_metres``
    (how far across one is) and ``canopy`` (the band of canopy light it grows
    in): what a plant *is* can be read off the model, but where it belongs in a
    wood is the world's business, not the scan's.

    ``keep`` takes only that many of the plants in the file, the fullest
    first: one published grass is seventeen tufts, and a field that grows all
    seventeen pays for seventeen scatters and sixty-eight draws to show what a
    handful of them already shows. Fullest by triangle count, which is how much
    plant there is -- the *tallest* tufts of a grass are its leggy seed stalks,
    and a card is the one thing that cannot show those.
    """
    from OpenGLContext.loaders.gltf.writer import (
        EncodedImage,
        GLTFWriter,
        SceneNode,
    )
    from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
    from OpenGLContext.scenegraph.pbrmesh import PBRMesh

    os.makedirs(directory, exist_ok=True)
    found = list(variants) if variants is not None else flatten(source.gltf)
    if keep is not None and keep < len(found):
        found = sorted(sorted(found, key=lambda one: one.name),
                       key=lambda one: -one.triangle_count)[:keep]
    texture = cutout(source.diffuse, source.mask)
    import io
    encoded = io.BytesIO()
    texture.save(encoded, format='PNG', optimize=True)

    material = PBRMaterial(alphaMode='MASK', alphaCutoff=ALPHA_CUTOFF,
                           doubleSided=True, metallic=0.0, roughness=0.9)
    # One image object for every rung of every variant: the writer keys its
    # image table on identity, so sharing the object is what stops a megabyte
    # of texture being written once per mesh.
    material.textures = {'baseColor': EncodedImage(
        encoded.getvalue(), mime_type='image/png', srgb=True)}

    writer = GLTFWriter(generator='OpenGLContext-editor plant bake')
    species: list[CoverSpecies] = []
    model = '%s.glb' % (source.slug,)
    for variant in found:
        near, far = _rungs(variant, list(rungs)[:2])
        for name, (attributes, indices) in (('%s' % (variant.name,), near),
                                            ('%s_far' % (variant.name,), far)):
            mesh = PBRMesh(positions=attributes['POSITION'],
                           normals=attributes['NORMAL'],
                           texcoords=attributes['TEXCOORD_0'],
                           indices=np.asarray(indices, 'u4').reshape(-1),
                           material=material)
            writer.add_mesh(mesh, name=name)
            writer.add_node(SceneNode(name=name, mesh=mesh))
        species.append(CoverSpecies(
            name=variant.name, card='%s_card.png' % (variant.name,),
            clump=model, clump_mesh=variant.name,
            clump_far_mesh='%s_far' % (variant.name,),
            density=density, height=round(variant.height, 4),
            patchiness=patchiness, canopy=canopy,
            **({} if patch_metres is None
               else {'patch_metres': patch_metres})))
    writer.write(os.path.join(directory, model))

    if card:
        # The card is the geometry rendered front-on, so how wide it came out
        # is how wide the plant is -- which is exactly what the billboard node
        # needs for its quad, and is not a figure anyone should be guessing.
        from dataclasses import replace

        from OpenGLContext_editor.assets.card import bake_card
        for index, (variant, entry) in enumerate(
                zip(found, species, strict=True)):
            width = bake_card(os.path.join(directory, model), variant.name,
                              texture, os.path.join(directory, entry.card),
                              size=card_size)
            species[index] = replace(entry, card_width=round(width, 4))
    _credit(directory, source)
    return species


def _credit(directory: str, source: PlantSource) -> None:
    """Add this plant's credit to the directory's own, once."""
    if not source.credit:
        return
    path = os.path.join(directory, 'CREDITS.txt')
    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            lines = handle.read().splitlines()
    if source.credit in lines:
        return
    lines.append(source.credit)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
