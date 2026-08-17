"""Axis-aligned bounds, and the frame change a 3D Tiles bounding volume needs.

The engine renders Y-up and 3D Tiles states a tile's frame Z-up, so the box a
tileset carries is not the box the geometry was measured in. These assert the
conversion in both directions, because a bake that gets it wrong produces tiles
whose contents are nowhere near where the traversal thinks they are.
"""

import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox


class TestABoundingBox:
    def test_it_measures_points(self) -> None:
        box = BoundingBox.of_points(np.array([(1, 2, 3), (-4, 5, 0)], "f"))
        assert box.minimum.tolist() == [-4, 2, 0]
        assert box.maximum.tolist() == [1, 5, 3]

    def test_an_empty_point_set_has_no_box(self) -> None:
        assert BoundingBox.of_points(np.zeros((0, 3), "f")) is None

    def test_it_reports_its_centre_and_half_extent(self) -> None:
        box = BoundingBox((-2, 0, -6), (2, 4, 6))
        assert box.center.tolist() == [0, 2, 0]
        assert box.half.tolist() == [2, 2, 6]

    def test_it_reports_its_diagonal(self) -> None:
        box = BoundingBox((0, 0, 0), (3, 0, 4))
        assert box.diagonal == pytest.approx(5.0)

    def test_boxes_join(self) -> None:
        joined = BoundingBox((0, 0, 0), (1, 1, 1)) | BoundingBox((-1, 2, 0), (0, 3, 1))
        assert joined.minimum.tolist() == [-1, 0, 0]
        assert joined.maximum.tolist() == [1, 3, 1]

    def test_joining_nothing_leaves_the_box_alone(self) -> None:
        box = BoundingBox((0, 0, 0), (1, 1, 1))
        assert (box | None) is box

    def test_it_knows_what_it_contains(self) -> None:
        box = BoundingBox((0, 0, 0), (2, 2, 2))
        assert box.contains((1, 1, 1))
        assert box.contains((0, 0, 0))          # the low corner is inside
        assert not box.contains((2.5, 1, 1))

    def test_it_expands(self) -> None:
        box = BoundingBox((0, 0, 0), (1, 1, 1)).expanded(0.5)
        assert box.minimum.tolist() == [-0.5, -0.5, -0.5]
        assert box.maximum.tolist() == [1.5, 1.5, 1.5]

    def test_a_flat_box_still_has_volume_to_a_traversal(self) -> None:
        """A perfectly flat tile (water, a road deck) must not collapse to a
        zero half-axis: the runtime's distance test would place the camera
        inside it at any height."""
        volume = BoundingBox((0, 5, 0), (10, 5, 10)).tiles_box()
        assert volume[11] > 0        # the Z half-axis, carrying the mesh's height
        assert (volume[3], volume[7]) == (5.0, 5.0)     # the other two are unharmed


class TestTheThreeDTilesFrame:
    def test_the_box_is_stated_z_up(self) -> None:
        """Y-up geometry, Z-up tile frame: the mesh's height becomes the box's
        Z extent and its depth becomes -Y."""
        box = BoundingBox((-10, 0, -30), (10, 4, 30))
        volume = box.tiles_box()
        center, x_axis, y_axis, z_axis = (volume[0:3], volume[3:6],
                                          volume[6:9], volume[9:12])
        assert center == pytest.approx([0.0, 0.0, 2.0])
        assert x_axis == pytest.approx([10.0, 0.0, 0.0])
        assert y_axis == pytest.approx([0.0, 30.0, 0.0])
        assert z_axis == pytest.approx([0.0, 0.0, 2.0])

    def test_it_round_trips_through_the_runtime(self) -> None:
        """What the runtime reads back has to be the box that went in."""
        from OpenGLContext.loaders.tiles3d.tileset import build_runtime_tileset
        box = BoundingBox((-8, 1, -20), (12, 9, 4))
        tileset = build_runtime_tileset(
            {'asset': {'version': '1.1'}, 'geometricError': 10.0,
             'root': {'boundingVolume': {'box': box.tiles_box()},
                      'geometricError': 5.0, 'refine': 'REPLACE'}},
            base_uri='', recenter=True)
        got = tileset.root.bounding_volume
        # recenter is how a viewer mounts a dataset that is not on the globe: it
        # turns the Z-up tile frame back into the Y-up world the box came from.
        assert np.allclose(got.center, box.center, atol=1e-9)
        assert got.bounding_sphere()[1] == pytest.approx(
            float(np.linalg.norm(box.half)))

    def test_a_sphere_volume_encloses_the_box(self) -> None:
        box = BoundingBox((0, 0, 0), (2, 2, 2))
        center, radius = box.tiles_sphere()[:3], box.tiles_sphere()[3]
        assert center == pytest.approx([1.0, -1.0, 1.0])
        assert radius == pytest.approx(np.sqrt(3.0))


class TestTheEverydayOperations:
    def test_it_prints_readably(self) -> None:
        assert 'BoundingBox' in repr(BoundingBox((0, 0, 0), (1, 2, 3)))

    def test_equal_boxes_compare_equal(self) -> None:
        assert BoundingBox((0, 0, 0), (1, 1, 1)) == BoundingBox((0, 0, 0), (1, 1, 1))
        assert BoundingBox((0, 0, 0), (1, 1, 1)) != BoundingBox((0, 0, 0), (2, 1, 1))

    def test_a_box_is_not_equal_to_something_else(self) -> None:
        assert BoundingBox((0, 0, 0), (1, 1, 1)) != 'a box'

    def test_boxes_can_key_a_dictionary(self) -> None:
        box = BoundingBox((0, 0, 0), (1, 1, 1))
        assert {box: 'tile'}[BoundingBox((0, 0, 0), (1, 1, 1))] == 'tile'

    def test_joining_a_sequence_of_none_gives_none(self) -> None:
        assert BoundingBox.joined([None, None]) is None

    def test_it_reports_its_footprint_size(self) -> None:
        assert BoundingBox((0, 0, 0), (3, 7, 5)).size.tolist() == [3, 7, 5]
