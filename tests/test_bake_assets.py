"""Authored glTF read back into meshes a bake can place.

The document is written with the engine's own writer, so what is asserted is
that a prototype survives the round trip *and* arrives flattened -- geometry in
one frame, transforms already applied, ready to be instanced.
"""

import math

import numpy as np
import pytest
from OpenGLContext.loaders.gltf.writer import SceneNode, write_glb
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

from OpenGLContext_editor.bake.assets import combined_mesh, meshes_from_gltf


def _triangle(**kwargs):
    kwargs.setdefault('positions', np.array([(0, 0, 0), (2, 0, 0), (0, 4, 0)], 'f'))
    kwargs.setdefault('normals', np.array([(0, 0, 1)] * 3, 'f'))
    kwargs.setdefault('indices', np.array([0, 1, 2], np.uint32))
    return PBRMesh(**kwargs)


class TestReadingADocument:
    def test_a_mesh_comes_back(self) -> None:
        meshes = meshes_from_gltf(write_glb(_triangle()))
        assert len(meshes) == 1
        assert np.allclose(meshes[0].positions, _triangle().positions, atol=1e-6)

    def test_its_material_comes_with_it(self) -> None:
        mesh = _triangle(material=PBRMaterial(baseColor=(0.1, 0.7, 0.2)))
        got = meshes_from_gltf(write_glb(mesh))[0]
        assert np.allclose(got.material.baseColor, (0.1, 0.7, 0.2), atol=1e-6)

    def test_several_meshes_all_come_back(self) -> None:
        document = write_glb([SceneNode(mesh=_triangle()),
                              SceneNode(mesh=_triangle(), translation=(9, 0, 0))])
        assert len(meshes_from_gltf(document)) == 2

    def test_a_node_translation_is_applied(self) -> None:
        document = write_glb(SceneNode(mesh=_triangle(), translation=(10, 1, -5)))
        got = meshes_from_gltf(document)[0]
        assert np.allclose(got.positions[0], (10, 1, -5), atol=1e-5)

    def test_a_node_scale_is_applied(self) -> None:
        document = write_glb(SceneNode(mesh=_triangle(), scale=(2, 3, 1)))
        got = meshes_from_gltf(document)[0]
        assert got.positions[:, 1].max() == pytest.approx(12.0, abs=1e-4)

    def test_a_node_rotation_is_applied(self) -> None:
        """A quarter turn about Y takes the +X corner to -Z."""
        half = math.sin(math.pi / 4), math.cos(math.pi / 4)
        document = write_glb(SceneNode(mesh=_triangle(),
                                       rotation=(0, half[0], 0, half[1])))
        got = meshes_from_gltf(document)[0]
        assert np.allclose(got.positions[1], (0, 0, -2), atol=1e-4)

    def test_nested_transforms_compose(self) -> None:
        child = SceneNode(mesh=_triangle(), translation=(1, 0, 0))
        parent = SceneNode(children=[child], translation=(0, 0, 10), scale=(2, 2, 2))
        got = meshes_from_gltf(write_glb(parent))[0]
        assert np.allclose(got.positions[0], (2, 0, 10), atol=1e-4)

    def test_normals_are_turned_with_the_geometry(self) -> None:
        """A quarter turn about +X takes a +Z normal to -Y."""
        half = math.sin(math.pi / 4), math.cos(math.pi / 4)
        document = write_glb(SceneNode(mesh=_triangle(),
                                       rotation=(half[0], 0, 0, half[1])))
        got = meshes_from_gltf(document)[0]
        assert np.allclose(got.normals[0], (0, -1, 0), atol=1e-4)

    def test_normals_stay_unit_length_under_a_stretch(self) -> None:
        document = write_glb(SceneNode(mesh=_triangle(), scale=(1, 8, 1)))
        got = meshes_from_gltf(document)[0]
        assert np.allclose(np.linalg.norm(got.normals, axis=1), 1.0, atol=1e-5)

    def test_a_prototype_can_be_rescaled_on_the_way_in(self) -> None:
        got = meshes_from_gltf(write_glb(_triangle()), scale=0.5)[0]
        assert got.positions[:, 1].max() == pytest.approx(2.0, abs=1e-5)

    def test_tangents_are_turned_too(self) -> None:
        half = math.sin(math.pi / 4), math.cos(math.pi / 4)
        mesh = _triangle(tangents=np.array([(1, 0, 0, 1)] * 3, 'f'),
                         texcoords=np.zeros((3, 2), 'f'))
        got = meshes_from_gltf(
            write_glb(SceneNode(mesh=mesh, rotation=(0, half[0], 0, half[1]))))[0]
        assert np.allclose(got.tangents[0][:3], (0, 0, -1), atol=1e-4)
        assert got.tangents[0][3] == pytest.approx(1.0)     # handedness is kept


class TestCombiningMeshes:
    def test_two_meshes_become_one(self) -> None:
        merged = combined_mesh([_triangle(), _triangle()])
        assert len(merged.positions) == 6
        assert merged.indices.tolist() == [0, 1, 2, 3, 4, 5]

    def test_a_non_indexed_mesh_gains_indices(self) -> None:
        merged = combined_mesh([_triangle(indices=None), _triangle()])
        assert merged.indices.tolist() == [0, 1, 2, 3, 4, 5]

    def test_one_shared_material_is_kept(self) -> None:
        material = PBRMaterial(baseColor=(1, 0, 0))
        merged = combined_mesh([_triangle(material=material),
                                _triangle(material=material)])
        assert merged.material is material

    def test_the_material_can_be_overridden(self) -> None:
        material = PBRMaterial(baseColor=(0, 1, 0))
        merged = combined_mesh([_triangle(), _triangle()], material=material)
        assert merged.material is material

    def test_a_mesh_without_colours_gets_white_ones(self) -> None:
        merged = combined_mesh([_triangle(), _triangle()])
        assert np.allclose(merged.colors, 1.0)

    def test_three_component_colours_gain_an_alpha(self) -> None:
        coloured = _triangle(colors=np.array([(1, 0, 0)] * 3, 'f'))
        merged = combined_mesh([coloured, _triangle()])
        assert merged.colors.shape == (6, 4)
        assert np.allclose(merged.colors[0], (1, 0, 0, 1))

    def test_combining_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match='no meshes'):
            combined_mesh([])

    def test_the_merged_mesh_still_round_trips(self) -> None:
        merged = combined_mesh([_triangle(), _triangle()])
        assert len(meshes_from_gltf(write_glb(merged))[0].positions) == 6


class TestReadingAVRMLTransform:
    """`_local_matrix` reads a VRML Transform, which has a `center` glTF lacks.

    Nothing the glTF loader builds sets one, so the case is reached here
    directly rather than through a written document.
    """

    def test_rotation_happens_about_the_centre(self) -> None:
        from OpenGLContext.scenegraph.transform import Transform

        from OpenGLContext_editor.bake.assets import _apply, _local_matrix
        node = Transform(center=(1, 0, 0), rotation=(0, 1, 0, math.pi))
        turned = _apply(np.array([[2.0, 0.0, 0.0]]), _local_matrix(node))
        # A half turn about (1,0,0) takes x=2 to x=0, not to x=-2.
        assert np.allclose(turned[0], (0, 0, 0), atol=1e-5)

    def test_a_plain_transform_is_the_identity(self) -> None:
        from OpenGLContext.scenegraph.transform import Transform

        from OpenGLContext_editor.bake.assets import _local_matrix
        assert np.allclose(_local_matrix(Transform()), np.identity(4))
