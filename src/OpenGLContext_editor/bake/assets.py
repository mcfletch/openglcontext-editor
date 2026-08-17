"""Reading authored geometry into a bake.

A world is not all procedural: trees, rocks, road furniture and buildings arrive
as glTF a designer made. :func:`meshes_from_gltf` loads such a document and
hands back the :class:`~OpenGLContext.scenegraph.pbrmesh.PBRMesh` objects inside
it, each one already moved into the document's own frame, so a prototype can be
dropped straight into an instance layer or a mesh layer.

Loading is the engine's job -- this reads through
:func:`OpenGLContext.loaders.gltf.load_gltf` -- and what is added here is the
flattening: the loaded scene is a tree of transforms, and a bake wants geometry
it can place itself.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from OpenGLContext.loaders import gltf
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

__all__ = ['meshes_from_gltf', 'combined_mesh']


def meshes_from_gltf(source: Any, scale: float = 1.0) -> list[PBRMesh]:
    """Every mesh in a glTF document, flattened into one frame.

    ``source`` is anything :func:`OpenGLContext.loaders.gltf.load_gltf` reads: a
    path, or the bytes of a ``.glb``. Each returned mesh keeps its own material
    and carries positions, normals and tangents already turned and moved by the
    transforms above it, with ``scale`` applied on top -- which is how a
    prototype authored at one size is planted at another.

    A node's ``scaleOrientation`` is not applied. Nothing the engine's own
    loader produces sets one, and a mesh that needs it is better exported
    without it than silently mis-shaped.
    """
    scene = gltf.load_gltf(source)
    matrix = np.identity(4, dtype='d')
    if scale != 1.0:
        matrix[0, 0] = matrix[1, 1] = matrix[2, 2] = float(scale)
    out: list[PBRMesh] = []
    _walk(scene.group, matrix, out)
    return out


def combined_mesh(meshes: list[PBRMesh], material: Any = None) -> PBRMesh:
    """One mesh from several, for content that need not keep them apart.

    Merging costs a draw call less per tile and, more to the point, lets a whole
    prototype ride one ``EXT_mesh_gpu_instancing`` node. The meshes must agree
    on which vertex attributes they carry; ``material`` overrides the result's,
    defaulting to the first mesh's.
    """
    if not meshes:
        raise ValueError("no meshes to combine")
    positions, normals, texcoords, colors, indices = [], [], [], [], []
    offset = 0
    for mesh in meshes:
        count = len(mesh.positions)
        positions.append(np.asarray(mesh.positions, 'f'))
        normals.append(_or_zeros(mesh.normals, count, 3))
        texcoords.append(_or_zeros(mesh.texcoords, count, 2))
        colors.append(_or_ones(mesh.colors, count))
        if mesh.indices is not None:
            indices.append(np.asarray(mesh.indices, np.uint32) + offset)
        else:
            indices.append(np.arange(count, dtype=np.uint32) + offset)
        offset += count
    return PBRMesh(
        positions=np.vstack(positions), normals=np.vstack(normals),
        texcoords=np.vstack(texcoords), colors=np.vstack(colors),
        indices=np.concatenate(indices),
        material=material if material is not None else meshes[0].material)


def _or_zeros(array: Any, count: int, width: int) -> np.ndarray:
    if array is None:
        return np.zeros((count, width), 'f')
    return np.asarray(array, 'f')


def _or_ones(array: Any, count: int) -> np.ndarray:
    if array is None:
        return np.ones((count, 4), 'f')
    values = np.asarray(array, 'f')
    if values.shape[1] == 3:
        values = np.hstack([values, np.ones((len(values), 1), 'f')])
    return values


def _walk(node: Any, matrix: np.ndarray, out: list[PBRMesh]) -> None:
    matrix = _local_matrix(node) @ matrix
    geometry = getattr(node, 'geometry', None)
    if isinstance(geometry, PBRMesh):
        out.append(_transformed(geometry, matrix))
    for child in getattr(node, 'children', None) or []:
        _walk(child, matrix, out)


def _transformed(mesh: PBRMesh, matrix: np.ndarray) -> PBRMesh:
    """A copy of a mesh with its geometry moved by a row-vector matrix."""
    if np.allclose(matrix, np.identity(4)):
        return mesh
    linear = matrix[:3, :3]
    positions = _apply(mesh.positions, matrix)
    # Normals follow the inverse transpose, so a non-uniform scale does not
    # tilt them off the surface.
    try:
        normal_matrix = np.linalg.inv(linear)
    except np.linalg.LinAlgError:                # pragma: no cover - degenerate scale
        normal_matrix = np.identity(3)
    normals = None
    if mesh.normals is not None:
        turned = np.asarray(mesh.normals, 'd') @ normal_matrix.T
        lengths = np.linalg.norm(turned, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        normals = (turned / lengths).astype('f')
    tangents = None
    if mesh.tangents is not None:
        source = np.asarray(mesh.tangents, 'd')
        turned = source[:, :3] @ linear
        lengths = np.linalg.norm(turned, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        tangents = np.hstack([turned / lengths, source[:, 3:4]]).astype('f')
    return PBRMesh(positions=positions, normals=normals, tangents=tangents,
                   texcoords=mesh.texcoords, texcoords1=mesh.texcoords1,
                   colors=mesh.colors, indices=mesh.indices,
                   material=mesh.material, draw_mode=mesh.draw_mode,
                   solid=mesh.solid)


def _apply(points: Any, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(points, 'd')
    homogeneous = np.hstack([array, np.ones((len(array), 1))])
    moved: np.ndarray = (homogeneous @ matrix)[:, :3].astype('f')
    return moved


def _local_matrix(node: Any) -> np.ndarray:
    """A VRML Transform's own matrix, row-vector (``p' = p @ M``)."""
    translation = _vector(getattr(node, 'translation', None))
    rotation = getattr(node, 'rotation', None)
    scale = _vector(getattr(node, 'scale', None), default=1.0)
    center = _vector(getattr(node, 'center', None))
    matrix = np.identity(4, dtype='d')
    if center is not None:
        matrix = matrix @ _translation(-center)
    if scale is not None:
        matrix = matrix @ np.diag([scale[0], scale[1], scale[2], 1.0])
    if rotation is not None and len(rotation) == 4 and float(rotation[3]):
        matrix = matrix @ _rotation(rotation)
    if center is not None:
        matrix = matrix @ _translation(center)
    if translation is not None:
        matrix = matrix @ _translation(translation)
    return matrix


def _vector(value: Any, default: float = 0.0) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype='d')
    if array.shape != (3,) or np.allclose(array, default):
        return None
    return array


def _translation(offset: np.ndarray) -> np.ndarray:
    matrix = np.identity(4, dtype='d')
    matrix[3, :3] = offset
    return matrix


def _rotation(axis_angle: Any) -> np.ndarray:
    """A VRML ``(x, y, z, radians)`` rotation as a row-vector matrix."""
    x, y, z, angle = (float(v) for v in axis_angle)
    length = math.sqrt(x * x + y * y + z * z)
    if not length:                               # pragma: no cover - no axis, no turn
        return np.identity(4, dtype='d')
    x, y, z = x / length, y / length, z / length
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    matrix = np.identity(4, dtype='d')
    matrix[:3, :3] = [
        [t * x * x + c, t * x * y + s * z, t * x * z - s * y],
        [t * x * y - s * z, t * y * y + c, t * y * z + s * x],
        [t * x * z + s * y, t * y * z - s * x, t * z * z + c],
    ]
    return matrix
