"""Levels of detail built from a mesh, and what they cost to look at.

A model arrives at whatever density its author left it, and a game cannot draw
that density at every distance. :mod:`~OpenGLContext_editor.meshlod.chain`
builds the coarser versions with :mod:`opengl_decimate`;
:mod:`~OpenGLContext_editor.meshlod.quality` says what each one looks like
against the original, which is the part that decides whether a level is usable
at all; and :mod:`~OpenGLContext_editor.meshlod.asset` writes them as the
``MSFT_lod`` glb the engine reads.

**This is a bake.** Decimating two hundred assets when a player opens a door is
not something that can be done, so the chain is made once, here, and what ships
is the result -- which the engine loads on its own, with none of this package
installed.

A level's *geometric* error is a length, and a length says nothing on its own:
a millimetre is invisible on a building and ruinous on a face. What matters is
what reaches the screen, so the measurement here is made by rendering -- the
fraction of the object's own pixels that change when a level is swapped in, at a
given distance. That number is what the switching distances are derived from.
"""

from OpenGLContext_editor.meshlod.asset import LODAsset, LODEntry, sidecar_name, write_chain
from OpenGLContext_editor.meshlod.chain import LODChain, LODLevel, build_chain
from OpenGLContext_editor.meshlod.quality import (
    LODProbe,
    measure_chain,
    object_pop,
    safe_distance,
    silhouette,
)

__all__ = [
    # Making the levels
    "build_chain",
    "LODChain",
    "LODLevel",
    # Keeping them on disk, and reading back only the one that is wanted
    "write_chain",
    "LODAsset",
    "LODEntry",
    "sidecar_name",
    # Finding out whether they are any good
    "LODProbe",
    "measure_chain",
    "object_pop",
    "safe_distance",
    "silhouette",
]
