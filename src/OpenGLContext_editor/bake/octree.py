"""The spatial partition a baked world is written along.

An octree rather than a quadtree, because a world with bridges, tunnels and
overhangs has content stacked vertically, and a height-surface partition cannot
place it. Where a world *is* a surface, the empty octants above and below it are
pruned and what remains is a quadtree in all but name, at no cost.

Two ways to build one:

:func:`uniform_octree`
    subdivide a region to a fixed depth, every node present. This is the
    terrain case, where the tree's shape follows the extent rather than the
    content.
:func:`build_octree`
    subdivide only where items are dense, pruning branches that hold nothing.
    This is the case for scattered content -- trees, props, road segments.

Both produce :class:`OctreeNode` trees satisfying what the 3D Tiles traversal
assumes: a child's bounds lie inside its parent's, every item is in exactly one
leaf, and the geometric error a node claims shrinks as the tree descends.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from OpenGLContext_editor.bake.bounds import BoundingBox

#: How many items a leaf holds before it splits, and how deep the split may go.
#: The depth limit is what stops coincident items -- a hundred trees planted at
#: one point -- from subdividing until the recursion gives out.
DEFAULT_MAX_ITEMS = 32
DEFAULT_MAX_DEPTH = 8


class OctreeNode:
    """One node: a region, the items inside it, and up to eight children.

    ``items`` holds indices into whatever array the tree was built over, and is
    populated on leaves only -- an interior node's items are its descendants'.
    ``payload`` is free for a caller to hang the node's baked content on.
    """

    __slots__ = ('bounds', 'level', 'children', 'items', 'payload')

    def __init__(self, bounds: BoundingBox, level: int = 0,
                 children: list[OctreeNode] | None = None,
                 items: Sequence[int] | None = None) -> None:
        self.bounds = bounds
        self.level = int(level)
        self.children: list[OctreeNode] = list(children or ())
        self.items: list[int] = list(items or ())
        self.payload: Any = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def octants(self, axes: Sequence[bool] = (True, True, True)
                ) -> list[OctreeNode]:
        """The children that tile this node's region exactly.

        ``axes`` selects which of X, Y and Z are halved, so a surface world can
        subdivide in the ground plane alone and spend no nodes on the air above
        it. All three gives the eight octants; two gives four; one gives a pair.
        """
        low, mid, high = self.bounds.minimum, self.bounds.center, self.bounds.maximum
        split = [axis for axis, wanted in enumerate(axes) if wanted]
        out = []
        for combination in range(1 << len(split)):
            corner_low, corner_high = list(low), list(high)
            for bit, axis in enumerate(split):
                if (combination >> bit) & 1:
                    corner_low[axis] = mid[axis]
                else:
                    corner_high[axis] = mid[axis]
            out.append(OctreeNode(BoundingBox(corner_low, corner_high),
                                  level=self.level + 1))
        return out

    def iter_nodes(self) -> Iterator[OctreeNode]:
        """This node and every node beneath it, parents before children."""
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def leaves(self) -> Iterator[OctreeNode]:
        for node in self.iter_nodes():
            if node.is_leaf:
                yield node

    def depth(self) -> int:
        """The deepest level anywhere in this subtree."""
        return max(node.level for node in self.iter_nodes())

    def geometric_error(self, root_error: float,
                        leaf_error: float | None = None) -> float:
        """The error this node's content stands in for, halving each level.

        A tile's geometric error is the world-space error a viewer accepts by
        drawing this node instead of its children, and each level covers half
        the ground at the same sampling rate. ``leaf_error`` overrides the value
        for a leaf, which is where a bake declares "nothing finer exists" with a
        zero.
        """
        if leaf_error is not None and self.is_leaf:
            return float(leaf_error)
        return float(root_error) / float(1 << self.level)

    def __repr__(self) -> str:
        return "OctreeNode(level=%d, items=%d, children=%d)" % (
            self.level, len(self.items), len(self.children))


def uniform_octree(bounds: BoundingBox, depth: int) -> OctreeNode:
    """A fully subdivided tree over ``bounds``, every node present to ``depth``."""
    root = OctreeNode(bounds, level=0)
    if depth > 0:
        root.children = [_grow_uniform(child, depth) for child in root.octants()]
    return root


def _grow_uniform(node: OctreeNode, depth: int) -> OctreeNode:
    if node.level < depth:
        node.children = [_grow_uniform(child, depth) for child in node.octants()]
    return node


def build_octree(positions: Any, bounds: BoundingBox | None = None,
                 max_items: int = DEFAULT_MAX_ITEMS,
                 max_depth: int = DEFAULT_MAX_DEPTH) -> OctreeNode:
    """Partition an (N,3) array of item positions, pruning empty branches.

    ``bounds`` defaults to the box enclosing the items, grown slightly so a point
    exactly on a face is still inside. The returned tree holds *indices* into
    ``positions``: what the item at each index is remains the caller's business.
    """
    points = np.asarray(positions, dtype='d').reshape(-1, 3)
    region = bounds or _default_bounds(points)
    root = OctreeNode(region, level=0, items=range(len(points)))
    _split(root, points, max_items, max_depth)
    return root


def _default_bounds(points: np.ndarray) -> BoundingBox:
    box = BoundingBox.of_points(points)
    if box is None:
        return BoundingBox((0, 0, 0), (1, 1, 1))
    # A cube, so the tree's octants stay cubical as it descends, and a margin so
    # a point on the maximum face is inside rather than on the boundary.
    half = float(max(box.half.max(), 0.5)) * 1.0001
    return BoundingBox.centred(box.center, (half, half, half))


def _split(node: OctreeNode, points: np.ndarray, max_items: int,
           max_depth: int) -> None:
    if len(node.items) <= max_items or node.level >= max_depth:
        return
    children = node.octants()
    # Which octant each item falls in: one bit per axis, above or below centre.
    index = np.asarray(node.items, dtype=np.intp)
    centre = node.bounds.center
    octant = np.zeros(len(index), dtype=np.intp)
    for axis in range(3):
        octant |= (points[index, axis] >= centre[axis]).astype(np.intp) << axis
    kept = []
    for slot, child in enumerate(children):
        child.items = index[octant == slot].tolist()
        if child.items:
            kept.append(child)
    if not kept:                    # pragma: no cover - every item is in an octant
        return
    node.children = kept
    node.items = []
    for child in kept:
        _split(child, points, max_items, max_depth)
