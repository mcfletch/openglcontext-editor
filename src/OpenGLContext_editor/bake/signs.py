"""A world's warning signs, written as one mesh and one picture per tile.

Every kind of plate reads out of one atlas
(:func:`OpenGLContext.scenegraph.roadsigns.sign_atlas`), and the post reads a
flat patch of its own colour from the same image, so a sign is one material
whatever it says. A tile's signs are then concatenated into a single mesh: one
render record, one bounding volume, one frustum test and one entry in each
shadow cascade for all of them.

Baked into place rather than instanced. Instancing is for thousands of copies of
one thing and buys a world with tens of signs nothing but the machinery -- and
to place a *different picture* per instance it would need a per-instance texture
offset, which is a fair amount of engine for a few hundred triangles a tile.

*Which* sign belongs *where* is decided in
:mod:`OpenGLContext_editor.world.signs`, from the road's own curvature and
grade. What is here is only the writing down.
"""
from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import ExternalImage, SceneNode
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.roadsigns import (
    SignProfile,
    atlas_material,
    sign_atlas,
    sign_mesh,
)

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.signs import Placement

#: Where a world keeps its sign artwork, relative to the tileset.
SIGN_DIRECTORY = 'signs'

#: The one image every plate in a world reads out of.
ATLAS_IMAGE = '%s/plates.png' % (SIGN_DIRECTORY,)

#: How big one plate is inside that image, in pixels.
PLATE_PIXELS = 256

#: The coarsest tile that carries signs. A sign is a metre across; a tile whose
#: error is tens of metres cannot show it and should not spend bandwidth on it.
MAXIMUM_ERROR = 12.0

#: How far a sign reaches beyond its own foot, in metres, for the bounds a tile
#: is chosen by. A post and its plate, with room for the turn.
SIGN_REACH = 1.2


@dataclass
class SignLayer:
    """Every warning sign in a world, as one mesh per tile.

    ``placements`` are what :func:`~OpenGLContext_editor.world.signs.warn_of`
    and :func:`~OpenGLContext_editor.world.signs.sign_placements` worked out.
    """

    placements: Sequence[Placement]
    profile: SignProfile = field(default_factory=SignProfile)
    plate_pixels: int = PLATE_PIXELS
    maximum_error: float = MAXIMUM_ERROR
    name: str = 'signs'
    _prototypes: dict = field(default_factory=dict, init=False, repr=False)
    _material: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        kinds = self.kinds()
        self._material = atlas_material(
            image=ExternalImage(ATLAS_IMAGE, srgb=True), kinds=kinds,
            cell=self.plate_pixels)
        cells = sign_atlas(kinds, self.plate_pixels)[1] if kinds else {}
        self._prototypes = {
            kind: sign_mesh(kind, self.profile, material=self._material,
                            cells=cells)
            for kind in kinds}

    def kinds(self) -> list[str]:
        """The kinds of sign this world actually has, in a settled order."""
        return sorted({one.kind for one in self.placements})

    def bounds(self) -> BoundingBox | None:
        """The ground the signs stand on, with their own height on it."""
        if not self.placements:
            return None
        feet = np.asarray([one.position for one in self.placements], dtype='d')
        box = BoundingBox.of_points(feet)
        assert box is not None
        top = float(self.profile.post_height + self.profile.plate_size)
        return BoundingBox(
            (box.minimum[0] - SIGN_REACH, box.minimum[1],
             box.minimum[2] - SIGN_REACH),
            (box.maximum[0] + SIGN_REACH, box.maximum[1] + top,
             box.maximum[2] + SIGN_REACH))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        if error > self.maximum_error:
            return []
        mine = [one for one in self.placements
                if _inside(one.position, region)]
        if not mine:
            return []
        return [SceneNode(mesh=_placed(mine, self._prototypes, self._material),
                          name=self.name)]

    def assets(self) -> dict[str, bytes]:
        """The one picture every plate in the world reads out of."""
        kinds = self.kinds()
        if not kinds:
            return {}
        buffer = io.BytesIO()
        sign_atlas(kinds, self.plate_pixels)[0].save(buffer, format='PNG')
        return {ATLAS_IMAGE: buffer.getvalue()}

    def metadata(self) -> dict[str, Any]:
        """Where the signs are, for a game that wants to read them.

        A driving aid, a minimap or an opponent that slows for a bend needs to
        know what the road says without going looking in the geometry for it.
        """
        return {'signs': [
            {'kind': one.kind,
             'at': [round(float(v), 3) for v in one.position],
             'yaw': round(float(one.yaw), 4)}
            for one in self.placements]}


def _placed(placements: Sequence[Placement], prototypes: dict,
            material: Any) -> PBRMesh:
    """A tile's signs, turned and moved into place, as one mesh."""
    positions, texcoords, normals, indices, offset = [], [], [], [], 0
    for one in placements:
        prototype = prototypes[one.kind]
        turn = _about_the_vertical(one.yaw)
        points = np.asarray(prototype.positions, dtype='d') @ turn.T
        positions.append(points + np.asarray(one.position, dtype='d'))
        normals.append(np.asarray(prototype.normals, dtype='d') @ turn.T)
        texcoords.append(np.asarray(prototype.texcoords))
        indices.append(np.asarray(prototype.indices) + offset)
        offset += len(prototype.positions)
    return PBRMesh(
        positions=np.concatenate(positions).astype('f'),
        normals=np.concatenate(normals).astype('f'),
        texcoords=np.concatenate(texcoords).astype('f'),
        indices=np.concatenate(indices).astype(np.uint32), material=material)


def _about_the_vertical(yaw: float) -> np.ndarray:
    """The rotation a sign is turned by to face the traffic it is for."""
    cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0],
                     [-sine, 0.0, cosine]])


def _inside(position: Any, region: BoundingBox) -> bool:
    at = np.asarray(position, dtype='d')
    return bool(np.all(at >= region.minimum) and np.all(at <= region.maximum))
