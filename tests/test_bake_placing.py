"""Standing a prototype somewhere, and writing several of them as one mesh."""
import numpy as np
import pytest
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

from OpenGLContext_editor.bake.placing import about_the_vertical, gathered, placed

MATERIAL = PBRMaterial(baseColor=(1.0, 1.0, 1.0))


def _flag():
    """A triangle standing on the origin, pointing along +X."""
    return PBRMesh(positions=np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                       (0.0, 1.0, 0.0)], dtype='f'),
                   normals=np.tile(np.array([0.0, 0.0, 1.0], 'f'), (3, 1)),
                   texcoords=np.zeros((3, 2), 'f'),
                   indices=np.array([0, 1, 2], dtype=np.uint32),
                   material=MATERIAL)


class TestTurning:
    def test_no_turn_leaves_it_alone(self) -> None:
        assert np.allclose(about_the_vertical(0.0), np.eye(3))

    def test_a_quarter_turn_puts_x_onto_minus_z(self) -> None:
        turned = about_the_vertical(np.pi / 2.0) @ np.array([1.0, 0.0, 0.0])
        assert np.allclose(turned, [0.0, 0.0, -1.0], atol=1e-9)

    def test_the_vertical_is_the_axis(self) -> None:
        turned = about_the_vertical(0.7) @ np.array([0.0, 1.0, 0.0])
        assert np.allclose(turned, [0.0, 1.0, 0.0])


class TestPlacing:
    def test_it_stands_where_it_is_put(self) -> None:
        stood = placed(_flag(), (10.0, 2.0, -3.0))
        assert np.allclose(stood.positions[0], [10.0, 2.0, -3.0])

    def test_it_faces_where_it_is_turned(self) -> None:
        stood = placed(_flag(), (0.0, 0.0, 0.0), yaw=np.pi / 2.0)
        assert np.allclose(stood.normals[0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_the_turn_reaches_the_geometry_too(self) -> None:
        stood = placed(_flag(), (0.0, 0.0, 0.0), yaw=np.pi / 2.0)
        assert np.allclose(stood.positions[1], [0.0, 0.0, -1.0], atol=1e-6)

    def test_it_keeps_the_picture_it_reads(self) -> None:
        prototype = _flag()
        assert placed(prototype, (1.0, 0.0, 0.0)).texcoords is prototype.texcoords


class TestGathering:
    def test_all_of_them_are_in_it(self) -> None:
        one = gathered([placed(_flag(), (0.0, 0.0, 0.0)),
                        placed(_flag(), (20.0, 0.0, 0.0))], MATERIAL)
        assert len(one.positions) == 6
        assert list(np.asarray(one.indices)) == [0, 1, 2, 3, 4, 5]

    def test_each_keeps_the_normals_it_was_placed_with(self) -> None:
        """Estimating them again would average across two separate objects."""
        one = gathered([placed(_flag(), (0.0, 0.0, 0.0)),
                        placed(_flag(), (0.0, 0.0, 0.0), yaw=np.pi / 2.0)],
                       MATERIAL)
        assert np.allclose(one.normals[0], [0.0, 0.0, 1.0], atol=1e-6)
        assert np.allclose(one.normals[3], [1.0, 0.0, 0.0], atol=1e-6)

    def test_it_wears_the_one_material(self) -> None:
        assert gathered([placed(_flag(), (0.0, 0.0, 0.0))],
                        MATERIAL).material is MATERIAL

    def test_gathering_nothing_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            gathered([], MATERIAL)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
