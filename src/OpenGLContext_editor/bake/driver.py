"""The bake: layers in, a streamable 3D Tiles world on disk out.

:func:`bake_world` walks a spatial partition of the world, asks every layer what
it has in each node at that node's geometric error, writes the answer as a
``.glb`` through the engine's glTF writer, and emits the ``tileset.json`` the
engine's 3D Tiles runtime streams.

**The partition subdivides in X and Z by default.** A world whose content sits on
a surface has one ground per column, and splitting the empty air above it buys
nothing but nodes. ``split_axes`` opens the third axis for a world that is
genuinely volumetric -- a city with levels, a cave system under a hillside.

**A tile's bounding volume is the box of what it actually wrote**, not the
partition cell it came from, unioned with its children's. That keeps the volume
tight enough for the screen-space-error test to be meaningful, and guarantees the
invariant the traversal rests on: a parent culled from the frustum has culled
every descendant with it.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import GLTFWriter, SceneNode

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.layers import Layer, node_bounds
from OpenGLContext_editor.bake.octree import OctreeNode
from OpenGLContext_editor.bake.tileset import BakedTile, write_tileset

#: Report progress after this many nodes when a caller gives no callback of its
#: own -- nothing is printed, this only bounds how often the callback fires.
ProgressFn = Callable[[int, int], None]

#: Subdivide in X and Z, leaving Y whole. See the module docstring.
SURFACE_AXES = (True, False, True)
VOLUME_AXES = (True, True, True)


@dataclass
class BakeResult:
    """What a bake produced, for a caller that wants to report or check it."""

    tileset: str
    directory: str
    tiles: int
    contents: int
    bytes_written: int
    root_error: float
    bounds: BoundingBox | None
    layers: dict[str, int] = field(default_factory=dict)
    assets: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One line a command-line bake can print."""
        return ("%d tiles, %d content files, %.1f MB, root error %.1f"
                % (self.tiles, self.contents, self.bytes_written / (1024 * 1024),
                   self.root_error))


def bake_world(layers: Sequence[Layer], directory: str,
               bounds: BoundingBox | None = None, depth: int = 3,
               root_error: float | None = None, leaf_error: float = 0.0,
               credits: Sequence[str] = (), name: str = 'tileset.json',
               prefix: str = 't', split_axes: Sequence[bool] = SURFACE_AXES,
               max_instances: int | None = None,
               progress: ProgressFn | None = None) -> BakeResult:
    """Bake ``layers`` into ``directory`` as a 3D Tiles world.

    ``bounds`` defaults to everything the layers cover. ``depth`` is how many
    times the partition subdivides; a surface world at depth *d* has
    ``(4**(d+1)-1)/3`` cells before empty ones are pruned. ``root_error``
    defaults to the root cell's horizontal size over the sampling rate, which is
    the scale at which drawing the root instead of its children is visibly wrong.

    ``max_instances`` overrides every instance layer's per-tile budget, for a
    caller tuning the whole bake at once rather than layer by layer.
    """
    region = bounds or BoundingBox.joined(layer.bounds() for layer in layers)
    if region is None:
        raise ValueError("nothing to bake: no layer covers any ground")
    if max_instances is not None:
        for layer in layers:
            if hasattr(layer, 'max_instances'):
                layer.max_instances = max_instances
    root_cell = _partition(region, depth, tuple(split_axes))
    os.makedirs(directory, exist_ok=True)
    shared = _write_assets(layers, directory)
    error = (root_error if root_error is not None
             else _default_root_error(region, layers))

    state = _BakeState(directory=directory, prefix=prefix,
                       total=len(list(root_cell.iter_nodes())), progress=progress)
    root_tile = _bake_node(root_cell, layers, error, leaf_error, state)
    extras: dict[str, Any] = {'bakedBy': 'OpenGLContext-editor'}
    for layer in layers:
        _contribute(extras, getattr(layer, 'metadata', dict)() or {}, layer)
    if root_tile is None:
        if not shared and len(extras) == 1:
            raise ValueError(
                "nothing to bake: no layer produced content anywhere in %r"
                % (region,))
        # A world whose ground is a field rather than tiles has geometry in no
        # tile at all, and is still a world: the landscape is written beside the
        # tileset and named from its extras. The root covers the region so a
        # viewer knows where the world is.
        root_tile = BakedTile(bounds=region, geometric_error=error)
    path = write_tileset(root_tile, directory, name=name, credits=credits,
                         extras=extras)
    return BakeResult(tileset=path, directory=directory,
                      tiles=len(list(root_tile.iter_tiles())),
                      contents=state.contents, bytes_written=state.bytes_written,
                      root_error=error, bounds=root_tile.bounds,
                      layers=dict(state.layer_counts), assets=shared)


def _contribute(extras: dict[str, Any], more: dict[str, Any],
                layer: Layer) -> None:
    """Fold one layer's metadata into the tileset's.

    Two layers may fill the same channel when it is a list: the props a gantry
    stands up and the boulders strewn over a landscape are one world's props,
    and a game reading them wants all of them. Two layers answering the same
    *question* is a bake to stop rather than to pick a winner from -- where a
    lap begins has one answer, and silently taking the last one writes a world
    whose timing belongs to whichever layer was listed later.
    """
    for key, value in more.items():
        held = extras.get(key)
        if key not in extras:
            extras[key] = value
        elif isinstance(held, list) and isinstance(value, list):
            extras[key] = held + value
        else:
            raise ValueError(
                "two layers disagree about %r: %s says %r and something "
                "before it said %r" % (key, layer.name, value, held))


def _write_assets(layers: Sequence[Layer], directory: str) -> list[str]:
    """Write the files layers share between tiles; return their names.

    A road surface, a decal atlas, a splat map: one file beside the tileset that
    every tile names, rather than a copy embedded in each of a thousand tiles.
    A name may be a relative path, so a layer with a set of files of its own --
    a forest's species -- keeps them in a directory rather than strewn beside
    the tileset.
    """
    written: list[str] = []
    for layer in layers:
        for name, data in (getattr(layer, 'assets', dict)() or {}).items():
            path = os.path.join(directory, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as handle:
                handle.write(data)
            written.append(name)
    return written


@dataclass
class _BakeState:
    """The running totals and naming a bake carries between nodes."""

    directory: str
    prefix: str
    total: int
    progress: ProgressFn | None
    visited: int = 0
    contents: int = 0
    bytes_written: int = 0
    layer_counts: dict[str, int] = field(default_factory=dict)

    def name_for(self, node: OctreeNode) -> str:
        centre = node.bounds.center
        return "%s_%d_%d_%d.glb" % (self.prefix, node.level,
                                    round(centre[0]), round(centre[2]))

    def step(self) -> None:
        self.visited += 1
        if self.progress is not None:
            self.progress(self.visited, self.total)


def _partition(region: BoundingBox, depth: int,
               axes: tuple[bool, ...]) -> OctreeNode:
    """A cell tree over ``region``, subdividing only along ``axes``."""
    root = OctreeNode(region, level=0)
    _grow(root, depth, axes)
    return root


def _grow(node: OctreeNode, depth: int, axes: tuple[bool, ...]) -> None:
    if node.level >= depth:
        return
    node.children = node.octants(axes=axes)
    for child in node.children:
        _grow(child, depth, axes)


def _default_root_error(region: BoundingBox, layers: Sequence[Layer]) -> float:
    """The root's error: how far apart its ground samples are, near enough.

    A terrain layer knows its own sampling rate, so the error is the root cell's
    width divided by it. With no such layer, the cell's diagonal stands in --
    something on the scale of the content, which is what the traversal compares
    against a screen-space tolerance.
    """
    width = float(max(region.size[0], region.size[2]))
    for layer in layers:
        resolution = getattr(layer, 'resolution', None)
        if resolution:
            return width / float(resolution) * 1.5
    return region.diagonal * 0.5


def _bake_node(cell: OctreeNode, layers: Sequence[Layer], root_error: float,
               leaf_error: float, state: _BakeState) -> BakedTile | None:
    """Bake one cell and its subtree, or None if nothing is there."""
    is_leaf = not cell.children
    error = float(leaf_error) if is_leaf else cell.geometric_error(root_error)
    nodes: list[SceneNode] = []
    for layer in layers:
        produced = layer.content(cell.bounds, error)
        if produced:
            nodes.extend(produced)
            state.layer_counts[layer.name] = state.layer_counts.get(layer.name, 0) + 1
    children = [child for child in
                (_bake_node(c, layers, root_error, leaf_error, state)
                 for c in cell.children) if child is not None]
    state.step()
    if not nodes and not children:
        return None

    content_uris: list[str] = []
    content_bounds: BoundingBox | None = None
    if nodes:
        writer = GLTFWriter()
        for node in nodes:
            writer.add_node(node)
        filename = state.name_for(cell)
        data = writer.write(os.path.join(state.directory, filename))
        state.contents += 1
        state.bytes_written += len(data)
        content_uris.append(filename)
        content_bounds = BoundingBox.joined(node_bounds(node) for node in nodes)

    bounds = BoundingBox.joined(
        [content_bounds] + [child.bounds for child in children]) or cell.bounds
    # A parent must not claim less error than a child, and a child of a
    # zero-error leaf would be impossible; the max keeps the ladder monotone
    # when a subtree turns out shallower than the partition.
    if children:
        error = max(error, max(child.geometric_error for child in children))
    return BakedTile(bounds=bounds, geometric_error=error,
                     content_uris=content_uris, children=children)


def bake_summary(result: BakeResult) -> str:
    """A short multi-line report of a bake, for a command line to print."""
    lines = [result.summary(), "  tileset: %s" % result.tileset]
    for name, count in sorted(result.layers.items()):
        lines.append("  %-12s %d tiles" % (name, count))
    if result.bounds is not None:
        lines.append("  extent:      %s to %s"
                     % (np.round(result.bounds.minimum, 1).tolist(),
                        np.round(result.bounds.maximum, 1).tolist()))
    return "\n".join(lines)
