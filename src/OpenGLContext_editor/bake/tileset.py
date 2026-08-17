"""Writing a ``tileset.json`` the engine's 3D Tiles runtime streams.

A :class:`BakedTile` tree in, an OGC 3D Tiles 1.1 document out.
:func:`tileset_document` produces the dict and :func:`write_tileset` puts it on
disk beside the ``.glb`` content its tiles name, together with the credits of
whatever data the world was baked from.

The document is written so a 1.0 reader can also load it: a tile with one
content uses ``content``, and only a tile carrying several uses the 1.1
``contents`` array.

The invariants the streaming traversal depends on are checked as the document is
built, because a tileset that breaks one of them does not fail -- it draws the
wrong tile, or nothing, at some camera position nobody tries until later:

* a tile's geometric error never exceeds its parent's, so refinement always
  buys detail;
* a child's bounding volume lies inside its parent's, so a parent culled from
  the frustum can be trusted to have culled its whole subtree.

Source: OGC 3D Tiles 1.1 — the tileset object, the tile object, ``refine``
inheritance, and ``geometricError``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from OpenGLContext_editor import __version__
from OpenGLContext_editor.bake.bounds import BoundingBox

GENERATOR = "OpenGLContext-editor %s" % __version__

#: The tolerance a child's bounds may exceed its parent's by. Bounds are
#: computed from float32 vertex data, so a child's box can miss its parent's by
#: a rounding step without anything being wrong.
BOUNDS_TOLERANCE = 1e-4

REFINE_REPLACE, REFINE_ADD = 'REPLACE', 'ADD'


@dataclass
class BakedTile:
    """One tile: what it covers, how wrong it is, and what to draw for it.

    ``geometric_error`` is the world-space error a viewer accepts by drawing
    this tile rather than refining into its children; zero means nothing finer
    exists. ``content_uris`` are relative to the tileset file, and a URI ending
    in ``.json`` is an external tileset the runtime grafts in as a subtree.
    """

    bounds: BoundingBox
    geometric_error: float
    content_uris: Sequence[str] = field(default_factory=list)
    children: Sequence[BakedTile] = field(default_factory=list)
    refine: str = REFINE_REPLACE
    transform: Sequence[float] | None = None

    def iter_tiles(self) -> Any:
        """This tile and every tile beneath it."""
        yield self
        for child in self.children:
            yield from child.iter_tiles()


def _tile_json(tile: BakedTile, inherited_refine: str | None) -> dict:
    if not np.isfinite(tile.geometric_error) or tile.geometric_error < 0:
        raise ValueError("a tile's geometric error must be finite and non-negative, "
                         "not %r" % (tile.geometric_error,))
    entry: dict[str, Any] = {
        'boundingVolume': {'box': tile.bounds.tiles_box()},
        'geometricError': float(tile.geometric_error),
    }
    if tile.refine != inherited_refine:
        entry['refine'] = tile.refine
    if tile.transform is not None:
        entry['transform'] = [float(v) for v in np.asarray(tile.transform).ravel()]
    uris = list(tile.content_uris)
    if len(uris) == 1:
        entry['content'] = {'uri': uris[0]}
    elif uris:
        entry['contents'] = [{'uri': uri} for uri in uris]
    if tile.children:
        for child in tile.children:
            _check_child(tile, child)
        entry['children'] = [_tile_json(child, tile.refine) for child in tile.children]
    return entry


def _check_child(parent: BakedTile, child: BakedTile) -> None:
    if child.geometric_error > parent.geometric_error + 1e-9:
        raise ValueError(
            "a child's geometric error (%g) exceeds its parent's (%g): refining "
            "into it would not buy detail" % (child.geometric_error,
                                              parent.geometric_error))
    outside = (np.any(child.bounds.minimum < parent.bounds.minimum - BOUNDS_TOLERANCE)
               or np.any(child.bounds.maximum > parent.bounds.maximum
                         + BOUNDS_TOLERANCE))
    if outside:
        raise ValueError(
            "a child's bounds fall outside its parent's (%r vs %r): culling the "
            "parent would cull content that is still in view"
            % (child.bounds, parent.bounds))


def tileset_document(root: BakedTile, geometric_error: float | None = None,
                     credits: Sequence[str] | None = None,
                     up_axis: str = 'Y', extras: dict | None = None) -> dict:
    """The tileset document for a tile tree.

    ``geometric_error`` is the error of drawing *nothing at all*, and so must
    exceed the root tile's or the root never loads; it defaults to twice the
    root's. ``up_axis`` states which axis the glTF content treats as up -- "Y"
    for everything the engine's writer produces.
    """
    asset: dict[str, Any] = {'version': '1.1', 'generator': GENERATOR,
                             'gltfUpAxis': up_axis}
    if credits:
        asset['copyright'] = '; '.join(credits)
    document: dict[str, Any] = {
        'asset': asset,
        'geometricError': float(geometric_error if geometric_error is not None
                                else max(root.geometric_error * 2.0, 1e-6)),
        'root': _tile_json(root, inherited_refine=None),
    }
    if extras:
        document['extras'] = extras
    return document


def write_tileset(root: BakedTile, directory: str, name: str = 'tileset.json',
                  geometric_error: float | None = None,
                  credits: Sequence[str] | None = None,
                  up_axis: str = 'Y', extras: dict | None = None) -> str:
    """Write a tileset (and its credits) into ``directory``; return its path.

    ``credits`` are the attributions the source data carries. They go into the
    document's ``asset.copyright`` and into a ``CREDITS.txt`` beside it, so a
    baked world can be shipped without losing the terms of the elevation and
    texture data it was made from.
    """
    os.makedirs(directory, exist_ok=True)
    document = tileset_document(root, geometric_error=geometric_error,
                                credits=credits, up_axis=up_axis, extras=extras)
    path = os.path.join(directory, name)
    with open(path, 'w') as handle:
        json.dump(document, handle, separators=(',', ':'))
    if credits:
        with open(os.path.join(directory, 'CREDITS.txt'), 'w') as handle:
            handle.write("Sources for the data this world was baked from.\n\n")
            for line in credits:
                handle.write("- %s\n" % line)
    return path
