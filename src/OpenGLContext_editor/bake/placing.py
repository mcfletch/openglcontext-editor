"""Turning a prototype into the world, and gathering the results into one mesh.

A world places tens of copies of a handful of prototypes -- signs, a gantry, a
line across the road. Each is modelled once at the origin and then stood
somewhere, facing somewhere, and the copies that land in one tile are written as
a single mesh: one render record, one bounding volume, one frustum test and one
entry in each shadow cascade for all of them.

Baked into place rather than instanced. Instancing is for thousands of copies of
one thing and buys a world with tens of them nothing but the machinery -- and
where the copies differ in the picture they carry, it would need a per-instance
texture offset, which is a fair amount of engine for a few hundred triangles.

:func:`gathered` keeps the normals it is given rather than estimating new ones.
The pieces are separate objects that happen to share a buffer, and a normal
averaged across two of them is a lit seam where they touch.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

__all__ = ['about_the_vertical', 'placed', 'gathered']


def about_the_vertical(yaw: float) -> np.ndarray:
    """The rotation that turns a prototype about the vertical by ``yaw``."""
    cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0],
                     [-sine, 0.0, cosine]])


def placed(mesh: PBRMesh, position: Any, yaw: float = 0.0) -> PBRMesh:
    """One prototype turned to ``yaw`` and stood at ``position``."""
    turn = about_the_vertical(yaw)
    points = np.asarray(mesh.positions, dtype='d') @ turn.T
    return PBRMesh(
        positions=(points + np.asarray(position, dtype='d')).astype('f'),
        normals=(np.asarray(mesh.normals, dtype='d') @ turn.T).astype('f'),
        texcoords=mesh.texcoords, indices=mesh.indices, material=mesh.material)


def gathered(meshes: Sequence[PBRMesh], material: Any) -> PBRMesh:
    """Several placed meshes of one material as a single mesh."""
    if not len(meshes):
        raise ValueError("nothing to gather: a mesh needs some geometry in it")
    positions, normals, texcoords, indices, offset = [], [], [], [], 0
    for mesh in meshes:
        positions.append(np.asarray(mesh.positions))
        normals.append(np.asarray(mesh.normals))
        texcoords.append(np.asarray(mesh.texcoords))
        indices.append(np.asarray(mesh.indices) + offset)
        offset += len(positions[-1])
    return PBRMesh(
        positions=np.concatenate(positions).astype('f'),
        normals=np.concatenate(normals).astype('f'),
        texcoords=np.concatenate(texcoords).astype('f'),
        indices=np.concatenate(indices).astype(np.uint32), material=material)
