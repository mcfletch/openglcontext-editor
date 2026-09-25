"""A published plant model, baked into ground cover the engine can grow.

What is published is authored for a render. What a field wants differs from
that in four ways:

One mesh per plant. A published file is a node per clump with the transform
that places it still to be applied -- often several plants in one file, which is
several kinds of cover for one download. :func:`flatten` applies the transforms
and hands back one :class:`Variant` per clump, stood at the origin on ``y=0``
where an instanced draw will put it.

The mask the model came without. These plants are alpha-cut cards, and the
published base colour is a JPEG, which cannot hold an alpha channel; the mask
ships as a map of its own. :func:`cutout` puts it back where the cutout shader
looks for it.

A triangle count a field can afford. A shrub authored at twenty-seven
thousand triangles is one shrub. :func:`bake` reduces each plant to two rungs --
what is drawn close up and what is drawn over the rest of the disc -- with
``opengl_decimate``, whose border handling is what keeps the silhouette of an
alpha card. One reduction serves both rungs, since each is a prefix of the same
recorded sequence of contractions.

One texture, not one per rung. A 1k RGBA cutout is about a megabyte, and a
plant of four variants at two rungs each would carry it eight times. So a bake
writes one ``.glb`` holding every rung of every variant against the single
embedded image they share, and each
:class:`~OpenGLContext.scenegraph.vegetation.cover.CoverSpecies` names the mesh
in it that is its own.

Normals are *carried* through the reduction rather than recomputed, so a
flattened triangle still shades the way the surface it replaced did; the same
holds in :mod:`OpenGLContext_editor.meshlod.chain`, which builds a whole chain
from the same reduction.
"""
from __future__ import annotations

import io
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
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

#: The primitive a triangle list is drawn with (``GL_TRIANGLES``), which is what
#: the loader turns strips and fans into.
GL_TRIANGLES = 4


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


def flatten(path: str) -> list[Variant]:
    """Every plant in a model file, transforms applied and stood at the origin.

    One :class:`Variant` per node that has triangles, its primitives merged --
    they share the one material, being one plant. Each is centred in plan on
    the middle of its bounding box and sat on ``y=0``, because an instanced
    draw supplies the position and the model's own is in the way of it.

    The file is read by the engine's glTF loader, so every storage form the
    format allows is read as the engine draws it, and every buffer and image
    the document names is resolved under the file's own directory.
    """
    from OpenGLContext.loaders.gltf import load_gltf, parse_gltf
    from OpenGLContext.loaders.gltf.animation import compute_world_matrices

    document = parse_gltf(path)
    scene = load_gltf(document=document)
    worlds = compute_world_matrices(scene.node_roots, scene.node_children,
                                    scene.node_transforms)
    nodes = document.gltf.nodes or []
    meshes = document.gltf.meshes or []
    found: list[Variant] = []

    def walk(index: int) -> None:
        node = nodes[index]
        if node.mesh is not None:
            named = (node.name or meshes[node.mesh].name
                     or 'part%d' % (index,))
            variant = _variant(_triangle_meshes(scene.node_transforms[index]),
                               worlds[index], _PUBLISHED_LOD.sub('', named))
            if variant is not None:
                found.append(variant)
        for child in scene.node_children.get(index, ()):
            walk(child)

    for root in scene.node_roots:
        walk(root)
    if not found:
        raise ValueError("%s holds no triangles to bake a plant from" % (path,))
    return found


def _triangle_meshes(group: Any) -> list[PBRMesh]:
    """The triangle meshes a glTF node's own primitives were built into."""
    return [child.geometry for child in getattr(group, 'children', ())
            if isinstance(getattr(child, 'geometry', None), PBRMesh)
            and child.geometry.draw_mode == GL_TRIANGLES]


def _variant(meshes: Sequence[PBRMesh], world: np.ndarray,
             name: str) -> Variant | None:
    """One node's triangle meshes, merged and placed by its row-vector
    ``world`` matrix (``p' = p @ world``)."""
    if not meshes:
        return None
    points, normals, uvs, indices, base = [], [], [], [], 0
    turn = world[:3, :3]
    # A normal is transformed by the inverse transpose, or a non-uniform scale
    # leaves it off the surface it belongs to.
    straighten = np.linalg.inv(turn).T
    for mesh in meshes:
        P = np.asarray(mesh.positions, 'd')
        points.append(P @ turn + world[3, :3])
        N = np.asarray(mesh.normals, 'd') @ straighten
        lengths = np.linalg.norm(N, axis=1, keepdims=True)
        normals.append(N / np.where(lengths > 1e-12, lengths, 1.0))
        uvs.append(np.zeros((len(P), 2), 'f4') if mesh.texcoords is None
                   else np.asarray(mesh.texcoords, 'f4'))
        drawn = (np.arange(len(P)) if mesh.indices is None
                 else np.asarray(mesh.indices).ravel())
        indices.append(drawn + base)
        base += len(P)
    P = np.concatenate(points)
    low, high = P.min(axis=0), P.max(axis=0)
    P = P - [(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2]
    return Variant(name=name, positions=P.astype('f4'),
                   normals=np.concatenate(normals).astype('f4'),
                   uvs=np.concatenate(uvs).astype('f4'),
                   indices=np.concatenate(indices).astype('u4'),
                   height=float(high[1] - low[1]))


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
    """The :class:`CoverSpecies` baked from ``source``, one per variant.

    Writes into ``directory`` one ``<slug>.glb`` holding every variant at every
    rung against the one cutout texture, the billboard each variant becomes
    past its geometry as ``<slug>_<variant>_card.png`` (with ``card``, which
    needs a GL context), and the credit for the work. A species names its card
    only where that file is in ``directory``: baked now, or by an earlier run.

    ``rungs`` is the triangle budget for the near and far geometry.
    ``density`` is plants per square metre, which the caller usually knows
    better than this does -- a grass is dense and a shrub is not. So are
    ``patchiness`` (how much the plant gathers into beds), ``patch_metres``
    (how far across one is) and ``canopy``, the band ``(least, most)`` of tree
    closure it grows under: 0 on open ground, 1 with a crown's worth of tree
    over every square metre, as
    :meth:`~OpenGLContext.scenegraph.terrain.splat.SplatTerrain.canopy_cover`
    measures it. What a plant *is* can be read off the model, but where it
    belongs in a wood is the world's business, not the scan's.

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

    os.makedirs(directory, exist_ok=True)
    found = list(variants) if variants is not None else flatten(source.gltf)
    if keep is not None and keep < len(found):
        found = sorted(sorted(found, key=lambda one: one.name),
                       key=lambda one: -one.triangle_count)[:keep]
    texture = cutout(source.diffuse, source.mask)
    encoded = io.BytesIO()
    texture.save(encoded, format='PNG', optimize=True)

    # One image object for every rung of every variant: the writer keys its
    # image table on identity, so sharing the object is what stops a megabyte
    # of texture being written once per mesh.
    material = PBRMaterial(alphaMode='MASK', alphaCutoff=ALPHA_CUTOFF,
                           doubleSided=True, metallic=0.0, roughness=0.9,
                           textures={'baseColor': EncodedImage(
                               encoded.getvalue(), mime_type='image/png',
                               srgb=True)})

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
            name=variant.name, clump=model, clumpMesh=variant.name,
            clumpFarMesh='%s_far' % (variant.name,),
            density=density, height=round(variant.height, 4),
            patchiness=patchiness, canopy=list(canopy or ()),
            **({} if patch_metres is None
               else {'patchMetres': patch_metres})))
    writer.write(os.path.join(directory, model))

    for index, (variant, entry) in enumerate(zip(found, species, strict=True)):
        named = '%s_%s_card.png' % (source.slug, variant.name)
        if card:
            # The card is the geometry rendered, so how wide it came out is how
            # wide the plant is: the figure the billboard node needs for its
            # quad.
            from OpenGLContext_editor.assets.card import bake_card
            width = bake_card(os.path.join(directory, model), variant.name,
                              texture, os.path.join(directory, named),
                              size=card_size)
            species[index] = entry.varied(card=named,
                                          cardWidth=round(width, 4))
        elif os.path.exists(os.path.join(directory, named)):
            species[index] = entry.varied(card=named)
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
