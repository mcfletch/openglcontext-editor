"""Levels of detail on disk, in a glB a normal glTF reader can still open.

Two hundred assets each carrying an ultra-high-resolution original cannot be
decimated when the player opens the door: the reduction is seconds per asset,
and it is the same answer every time. It is baked once, and what ships is the
chain. The question is then what the chain looks like on disk, and the answer
has to satisfy two things at once -- a finer level must not cost memory,
bandwidth or parse time until something asks for it, and the file should still
be a file other tools understand.

glTF answers both, without an invention:

A glB may point outside itself. The binary chunk inside a glB is buffer
zero and is the one buffer with no ``uri``. Every *other* buffer is an ordinary
glTF buffer and may name an external file. So the coarse levels live in the
chunk, always there and always cheap, and each finer level is its own sidecar
that the operating system never opens until it is wanted.

glTF is addressable. Every array reaches its bytes through
``accessor -> bufferView -> buffer``, and a bufferView is an offset and a
length. There is no container to walk, no compression spanning the file and no
need to read what comes before: a level's vertices are a known byte range in a
known file, and reading it is a seek and a read. The one thing that must be
parsed is the JSON chunk, which is kilobytes.

The levels are declared the way the ecosystem declares them, with
``MSFT_lod``: the node carrying the extension is the finest, ``ids`` lists the
coarser alternatives in decreasing detail, and ``MSFT_screencoverage`` in
``extras`` says at what share of the screen each takes over. A reader that does
not know the extension ignores it and draws the finest level, which is the
correct thing for it to do; a reader that does know it picks a level.

    from OpenGLContext_editor.meshlod import write_chain, LODAsset

    write_chain('bust.glb', chain)          # bust.glb + bust.lod0.bin, ...

    asset = LODAsset.open('bust.glb')       # parses kilobytes of JSON
    asset.levels[-1].triangle_count         # the coarsest, in the glb itself
    asset.load(len(asset.levels) - 1)       # a seek and a read

The file is written by the engine's
:meth:`~OpenGLContext.loaders.gltf.writer.GLTFWriter.add_lod`, and
:class:`~OpenGLContext.loaders.gltf.lodasset.LODAsset` is the engine's reader,
named here as well; what this module adds is the layout -- which levels ride in
the glb, what the sidecars are called -- and the chain's recorded errors.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.lodasset import LOD_ERROR, LODAsset, LODEntry
from OpenGLContext.loaders.gltf.writer import GLTFWriter, SceneNode
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

__all__ = ["LODAsset", "LODEntry", "write_chain", "sidecar_name"]

#: Which chain attribute becomes which :class:`PBRMesh` array.
_ARRAYS = {
    "POSITION": "positions",
    "NORMAL": "normals",
    "TANGENT": "tangents",
    "TEXCOORD_0": "texcoords",
    "TEXCOORD_1": "texcoords1",
    "COLOR_0": "colors",
}


def sidecar_name(path: str, level: int) -> str:
    """Where level ``level``'s geometry lives beside ``path``."""
    stem, _ = os.path.splitext(path)
    return "%s.lod%d.bin" % (stem, level)


def _mesh(level: Any) -> PBRMesh:
    """One level of a chain as the mesh the writer takes."""
    arrays = {
        field: np.ascontiguousarray(level.attributes[name], dtype="<f4")
        for name, field in _ARRAYS.items()
        if name in level.attributes
    }
    return PBRMesh(indices=np.ascontiguousarray(level.indices, dtype="<u4").reshape(-1),
                   **arrays)


def write_chain(
    path: str,
    chain: Any,
    coverage: Sequence[float] | None = None,
    embed_coarsest: int = 1,
) -> list[str]:
    """The paths written, the glb first: a chain as a glb plus one sidecar per
    level kept out of it.

    ``embed_coarsest`` levels go into the glb's own binary chunk -- they are the
    ones always wanted, and a file that needs no sidecar to draw something is a
    file that always draws something. Every finer level becomes its own
    ``.lodN.bin``, so the cost of having it is a directory entry until it is
    asked for.

    ``coverage`` is the ``MSFT_screencoverage`` list, finest first; without one
    the series the engine's loader would guess is written -- halving, ending at
    0 so the coarsest level is never culled
    (:func:`OpenGLContext.loaders.gltf.lod.halving_coverage`).
    """
    levels = list(chain)
    if not levels:
        raise ValueError("a chain with no levels cannot be written")
    finer = len(levels) - max(0, int(embed_coarsest))
    nodes = [
        SceneNode(
            mesh=_mesh(level),
            name="lod%d" % (index,),
            buffer=os.path.basename(sidecar_name(path, index)) if index < finer else None,
        )
        for index, level in enumerate(levels)
    ]
    writer = GLTFWriter(generator="OpenGLContext_editor.meshlod")
    writer.add_lod(nodes, coverage)
    writer.extras[LOD_ERROR] = [float(level.error) for level in levels]
    writer.write(path)
    directory = os.path.dirname(path) or "."
    return [path] + [os.path.join(directory, name) for name in writer.external_buffers()]
