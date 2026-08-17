"""The spatial partition: invariants a baked tree has to hold.

Every item lands in exactly one leaf, every child's bounds lie inside its
parent's, and the error a node claims shrinks as the tree descends. Those three
are what the streaming traversal assumes; a tree that breaks any of them draws
the wrong tile or none at all.
"""

import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.octree import OctreeNode, build_octree, uniform_octree


def _grid_points(n=4, extent=100.0):
    axis = np.linspace(-extent, extent, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)


class TestSubdivision:
    def test_a_node_splits_into_eight(self) -> None:
        node = OctreeNode(BoundingBox((-1, -1, -1), (1, 1, 1)), level=0)
        octants = node.octants()
        assert len(octants) == 8
        assert all(o.level == 1 for o in octants)

    def test_the_octants_tile_the_parent_exactly(self) -> None:
        parent = BoundingBox((-2, 0, 4), (6, 8, 20))
        octants = OctreeNode(parent, level=0).octants()
        joined = BoundingBox.joined(o.bounds for o in octants)
        assert np.allclose(joined.minimum, parent.minimum)
        assert np.allclose(joined.maximum, parent.maximum)
        volume = sum(float(np.prod(o.bounds.size)) for o in octants)
        assert volume == pytest.approx(float(np.prod(parent.size)))

    def test_octants_do_not_overlap(self) -> None:
        octants = OctreeNode(BoundingBox((0, 0, 0), (2, 2, 2)), level=0).octants()
        centres = [tuple(o.bounds.center) for o in octants]
        assert len(set(centres)) == 8


class TestAUniformTree:
    def test_depth_gives_the_expected_node_count(self) -> None:
        root = uniform_octree(BoundingBox((0, 0, 0), (8, 8, 8)), depth=2)
        assert len(list(root.iter_nodes())) == 1 + 8 + 64

    def test_leaves_are_at_the_requested_depth(self) -> None:
        root = uniform_octree(BoundingBox((0, 0, 0), (8, 8, 8)), depth=3)
        assert {leaf.level for leaf in root.leaves()} == {3}

    def test_a_depth_of_zero_is_a_single_node(self) -> None:
        root = uniform_octree(BoundingBox((0, 0, 0), (1, 1, 1)), depth=0)
        assert not root.children and root.is_leaf


class TestAnAdaptiveTree:
    def test_every_item_lands_in_exactly_one_leaf(self) -> None:
        points = _grid_points(5)
        root = build_octree(points, max_items=4, max_depth=6)
        landed = [index for leaf in root.leaves() for index in leaf.items]
        assert sorted(landed) == list(range(len(points)))

    def test_an_item_lands_in_a_leaf_that_contains_it(self) -> None:
        points = _grid_points(4)
        root = build_octree(points, max_items=3, max_depth=6)
        for leaf in root.leaves():
            for index in leaf.items:
                assert leaf.bounds.contains(points[index]), index

    def test_leaves_hold_no_more_than_the_limit(self) -> None:
        root = build_octree(_grid_points(5), max_items=8, max_depth=8)
        assert all(len(leaf.items) <= 8 for leaf in root.leaves())

    def test_depth_bounds_the_split_even_when_items_pile_up(self) -> None:
        """Coincident points cannot be separated; the tree must stop anyway."""
        points = np.zeros((50, 3))
        root = build_octree(points, max_items=2, max_depth=3)
        assert max(node.level for node in root.iter_nodes()) <= 3
        assert sum(len(leaf.items) for leaf in root.leaves()) == 50

    def test_empty_branches_are_pruned(self) -> None:
        """One cluster in a corner must not cost eight children per level."""
        points = np.random.default_rng(4).normal(size=(40, 3)) * 0.5 + 40.0
        root = build_octree(points, bounds=BoundingBox((-64, -64, -64), (64, 64, 64)),
                            max_items=4, max_depth=4)
        assert all(node.items or node.children for node in root.iter_nodes())
        assert len(list(root.iter_nodes())) < 1 + 8 + 64 + 512

    def test_the_root_covers_the_items_when_no_bounds_are_given(self) -> None:
        points = _grid_points(3, extent=7.0)
        root = build_octree(points, max_items=4)
        assert all(root.bounds.contains(p) for p in points)

    def test_no_items_makes_an_empty_root(self) -> None:
        root = build_octree(np.zeros((0, 3)), bounds=BoundingBox((0, 0, 0), (1, 1, 1)))
        assert root.is_leaf and not root.items


class TestChildBoundsAndError:
    def test_child_bounds_lie_inside_the_parent(self) -> None:
        root = build_octree(_grid_points(5), max_items=3, max_depth=5)
        for node in root.iter_nodes():
            for child in node.children:
                assert np.all(child.bounds.minimum >= node.bounds.minimum - 1e-9)
                assert np.all(child.bounds.maximum <= node.bounds.maximum + 1e-9)

    def test_geometric_error_halves_with_each_level(self) -> None:
        root = uniform_octree(BoundingBox((0, 0, 0), (64, 64, 64)), depth=3)
        errors = {node.level: node.geometric_error(root_error=32.0)
                  for node in root.iter_nodes()}
        assert errors == {0: 32.0, 1: 16.0, 2: 8.0, 3: 4.0}

    def test_a_leaf_can_be_told_to_claim_no_error(self) -> None:
        """A leaf with nothing finer beneath it refines no further."""
        root = uniform_octree(BoundingBox((0, 0, 0), (8, 8, 8)), depth=1)
        leaf = next(iter(root.leaves()))
        assert leaf.geometric_error(root_error=16.0, leaf_error=0.0) == 0.0

    def test_content_scale_follows_the_node_size(self) -> None:
        """The natural error for a node is its own size over the sampling rate."""
        root = uniform_octree(BoundingBox((0, 0, 0), (128, 128, 128)), depth=2)
        for node in root.iter_nodes():
            assert node.geometric_error(root_error=64.0) == pytest.approx(
                64.0 / (2 ** node.level))


class TestTheEverydayOperations:
    def test_a_node_prints_readably(self) -> None:
        node = OctreeNode(BoundingBox((0, 0, 0), (1, 1, 1)), level=2)
        assert 'level=2' in repr(node)

    def test_a_tree_reports_its_depth(self) -> None:
        assert uniform_octree(BoundingBox((0, 0, 0), (8, 8, 8)), depth=2).depth() == 2

    def test_an_empty_item_set_still_gives_a_usable_region(self) -> None:
        """A layer with nothing placed yet must not produce a degenerate box."""
        root = build_octree(np.zeros((0, 3)))
        assert np.all(root.bounds.size > 0)

    def test_only_two_axes_split_into_four(self) -> None:
        node = OctreeNode(BoundingBox((0, 0, 0), (2, 2, 2)), level=0)
        quads = node.octants(axes=(True, False, True))
        assert len(quads) == 4
        assert all(q.bounds.size[1] == 2 for q in quads)     # Y is left whole
