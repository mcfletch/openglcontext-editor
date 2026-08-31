"""What a baked world is, written down beside the tiles that draw it.

The format and the reading of it are the engine's
(:mod:`OpenGLContext.loaders.tiles3d.manifest`), because anything that loads a
baked world wants to know what it is and most of those are not authoring tools.
What is here is the name the bake reaches for.
"""
from __future__ import annotations

import os

from OpenGLContext.loaders.tiles3d.manifest import (
    MANIFEST,
    WorldManifest,
    carried,
    read_manifest,
    write_manifest,
)

__all__ = ['MANIFEST', 'WorldManifest', 'carried', 'read_manifest',
           'world_name', 'write_manifest']


def world_name(directory: str) -> str:
    """A readable name for a world baked into that directory.

    A world baked into ``ashdown-forest`` is *Ashdown Forest* until somebody
    says otherwise, which beats making every world "Untitled" and beats asking
    for a name before one can be baked at all.
    """
    stem = os.path.basename(os.path.abspath(directory))
    return stem.replace('-', ' ').replace('_', ' ').strip().title() or 'Untitled'
