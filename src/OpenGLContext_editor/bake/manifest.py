"""What a baked world is, written down beside the tiles that draw it.

The format and the reading of it are the engine's
(:mod:`OpenGLContext.loaders.tiles3d.manifest`), because anything that loads a
baked world wants to know what it is and most of those are not authoring tools.
What is here is the name the bake reaches for.
"""
from __future__ import annotations

from OpenGLContext.loaders.tiles3d.manifest import (
    MANIFEST,
    WorldManifest,
    carried,
    read_manifest,
    write_manifest,
)

__all__ = ['MANIFEST', 'WorldManifest', 'carried', 'read_manifest',
           'write_manifest']
