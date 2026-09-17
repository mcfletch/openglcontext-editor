"""Turning published art into the assets a world is baked with.

A world needs plants, and good ones are published rather than modelled: Poly
Haven scans them and releases them under CC0, which is compatible with the terms
everything here ships under and asks for no attribution (it is given anyway).
What is published is authored for a render, though, and a field wants something
else entirely -- so this is the gap between the two.

- :mod:`~OpenGLContext_editor.assets.polyhaven` fetches a model and the maps it
  needs.
- :mod:`~OpenGLContext_editor.assets.plants` bakes one into the ground cover the
  engine grows: one mesh per plant, two levels of detail, and a single cutout
  texture.
- :mod:`~OpenGLContext_editor.assets.card` renders the billboard that stands in
  for a plant past the distance its geometry is worth drawing.
- :mod:`~OpenGLContext_editor.assets.rewrap` gives a reduced mesh an atlas of
  its own and bakes the original's texture into it, for a scan whose published
  unwrap is finer than the levels made from it.

This is authoring. An editor imports it; a shipped game has the baked files and
needs none of it.
"""
from OpenGLContext_editor.assets.plants import (
    PlantSource,
    Variant,
    bake,
    cutout,
    flatten,
)

__all__ = ['PlantSource', 'Variant', 'bake', 'cutout', 'flatten']
