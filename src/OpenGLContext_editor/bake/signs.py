"""A world's warning signs, baked as one prototype per kind.

A world has tens of signs and a handful of kinds, so each kind is written as one
instanced node -- the post, and the plate wearing that kind's picture -- placed
wherever a sign of that kind stands. The picture itself is written once beside
the tileset and named by every tile carrying that kind, because a plate embedded
per tile arrives again with every tile.

*Which* sign belongs *where* is decided in
:mod:`OpenGLContext_editor.world.signs`, from the road's own curvature and
grade; the object is the engine's
(:mod:`OpenGLContext.scenegraph.roadsigns`). What is here is only the writing
down.
"""
from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import ExternalImage, InstanceSet, SceneNode
from OpenGLContext.scenegraph.roadsigns import (
    SignProfile,
    post_material,
    sign_material,
    sign_meshes,
    sign_texture,
)

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.signs import Placement

#: Where a world keeps its plates, relative to the tileset.
SIGN_DIRECTORY = 'signs'

#: How big a plate's picture is written, in pixels.
PLATE_PIXELS = 256

#: The coarsest tile that carries signs. A sign is a metre across; a tile whose
#: error is tens of metres cannot show it and should not spend bandwidth on it.
MAXIMUM_ERROR = 12.0

#: How far a sign reaches beyond its own foot, in metres, for the bounds a tile
#: is chosen by. A post and its plate, with room for the turn.
SIGN_REACH = 1.2


@dataclass
class SignLayer:
    """Every warning sign in a world, as one instanced node per kind and part.

    ``placements`` are what :func:`~OpenGLContext_editor.world.signs.warn_of`
    and :func:`~OpenGLContext_editor.world.signs.sign_placements` worked out.
    """

    placements: Sequence[Placement]
    profile: SignProfile = field(default_factory=SignProfile)
    plate_pixels: int = PLATE_PIXELS
    maximum_error: float = MAXIMUM_ERROR
    name: str = 'signs'
    _prototypes: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._prototypes = {
            kind: sign_meshes(kind, self.profile,
                              material=sign_material(
                                  kind, image=ExternalImage(
                                      self._plate_name(kind), srgb=True)),
                              post=post_material())
            for kind in self.kinds()}

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
        found: list[SceneNode] = []
        for kind in self.kinds():
            mine = [one for one in self.placements
                    if one.kind == kind and _inside(one.position, region)]
            if not mine:
                continue
            instances = InstanceSet(
                translations=np.asarray([one.position for one in mine], 'f'),
                rotations=_yaws([one.yaw for one in mine]))
            found.extend(
                SceneNode(mesh=mesh, instances=instances,
                          name='%s-%s-%s' % (self.name, kind, part))
                for part, mesh in sorted(self._prototypes[kind].items()))
        return found

    def assets(self) -> dict[str, bytes]:
        """One plate picture per kind, written beside the tileset."""
        written = {}
        for kind in self.kinds():
            buffer = io.BytesIO()
            sign_texture(kind, self.plate_pixels).save(buffer, format='PNG')
            written[self._plate_name(kind)] = buffer.getvalue()
        return written

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

    def _plate_name(self, kind: str) -> str:
        return '%s/%s.png' % (SIGN_DIRECTORY, kind)


def _inside(position: Any, region: BoundingBox) -> bool:
    at = np.asarray(position, dtype='d')
    return bool(np.all(at >= region.minimum) and np.all(at <= region.maximum))


def _yaws(angles: Sequence[float]) -> np.ndarray:
    """Rotations about the vertical, as the quaternions glTF instancing wants."""
    half = np.asarray(angles, dtype='d') / 2.0
    return np.stack([np.zeros(len(half)), np.sin(half), np.zeros(len(half)),
                     np.cos(half)], axis=-1).astype('f')
