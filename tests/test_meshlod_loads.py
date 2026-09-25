"""A chain this package bakes is a chain the engine draws.

The two halves are in different distributions -- the levels are made here and
switched between there -- so the seam between them is a file, and this is the
case that holds it: write a chain, load the glb through the engine's glTF
loader, and find the levels on the ``ScreenCoverageLOD`` it built. What this
catches is a change to either half that leaves each one's own suite green.

The finer levels live in sidecars beside the glb, so loading them back is also
what says the sidecars were written where the file says they are.
"""

import numpy as np
import pytest
from OpenGLContext.loaders import gltf
from OpenGLContext.scenegraph.lod import ScreenCoverageLOD
from OpenGLContext.scenegraph.shape import Shape

from OpenGLContext_editor.meshlod import write_chain
from OpenGLContext_editor.meshlod.chain import LODChain, LODLevel


def _triangle(scale=1.0):
    return np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype='f4') * scale


def _flatten(node, out=None):
    out = [] if out is None else out
    out.append(node)
    for child in getattr(node, 'children', None) or []:
        _flatten(child, out)
    for level in getattr(node, 'level', None) or []:
        _flatten(level, out)
    return out


def _the_lod(scene):
    found = [n for n in _flatten(scene.group) if isinstance(n, ScreenCoverageLOD)]
    assert len(found) == 1, found
    return found[0]


def _sizes(node):
    return [max(n.geometry.positions.max() for n in _flatten(level)
                if isinstance(n, Shape))
            for level in node.level]


@pytest.fixture
def written(tmp_path):
    """Three levels of one triangle, baked to a glb and its sidecars."""
    levels = [
        LODLevel(
            attributes={'POSITION': _triangle(1.0 / (index + 1)),
                        'NORMAL': np.tile(np.array([0, 0, 1], 'f4'), (3, 1))},
            indices=np.array([0, 1, 2], np.uint32),
            error=0.01 * index,
            vertex_map=np.arange(3, dtype=np.uint32),
        )
        for index in range(3)
    ]
    chain = LODChain(levels=levels, centre=(0.5, 0.5, 0.0), radius=1.0)
    return write_chain(str(tmp_path / 'bust.glb'), chain)


class TestWhatTheEngineMakesOfABakedChain:
    def test_every_level_is_there(self, written):
        assert len(_the_lod(gltf.load_gltf(written[0])).level) == 3

    def test_the_levels_are_in_decreasing_detail(self, written):
        assert _sizes(_the_lod(gltf.load_gltf(written[0]))) == pytest.approx(
            [1.0, 0.5, 1.0 / 3.0])

    def test_the_coverage_written_is_the_coverage_read(self, written):
        node = _the_lod(gltf.load_gltf(written[0]))

        assert list(node.screenCoverage) == pytest.approx([0.5, 0.25, 0.0])

    def test_the_object_is_sized_without_its_geometry_being_read(self, written):
        """From the POSITION accessor bounds the writer declares."""
        node = _the_lod(gltf.load_gltf(written[0]))

        assert node.coverageRadius() == pytest.approx(np.sqrt(2.0) / 2.0)
