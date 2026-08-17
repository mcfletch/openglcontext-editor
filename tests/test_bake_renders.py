"""A baked world, streamed back through the engine's viewer and looked at.

Everything else in this suite asserts numbers. This one bakes a world, runs
``oglc-view`` over it in a subprocess exactly as a user would, and checks the
frame that comes out: ground where ground belongs, sky above it, and the trees
present as their own colour. A tileset can be numerically perfect and still
arrive inside out, and nothing but a render says so.

The viewer renders offscreen (``OPENGLCONTEXT_HIDDEN``), so this needs a GPU but
not a display.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from OpenGLContext_editor.bake.driver import bake_world
from OpenGLContext_editor.world.procedural import ProceduralWorld

pytest.importorskip('PIL')


def _viewer():
    """``oglc-view``, looked for beside the interpreter before on PATH.

    A suite run as ``.venv/bin/python -m pytest`` has the venv's scripts next to
    ``sys.executable`` but not necessarily on PATH, and a GL test that skips
    itself reads as green while rendering nothing.
    """
    beside = os.path.join(os.path.dirname(sys.executable), 'oglc-view')
    if os.path.exists(beside):
        return beside
    return shutil.which('oglc-view')


VIEWER = _viewer()
pytestmark = pytest.mark.skipif(
    VIEWER is None, reason="oglc-view is not installed; nothing to render with")

#: A small world: enough tiles to stream, quick enough for a suite.
EXTENT = 1024.0
DEPTH = 2


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    """Bake a world, render one frame of it from inside, return the pixels."""
    from PIL import Image
    directory = tmp_path_factory.mktemp('rendered')
    world = ProceduralWorld(extent=EXTENT, resolution=17, seed=11)
    result = bake_world(world.layers(), str(directory), depth=DEPTH)
    capture = os.path.join(str(directory), 'view.png')
    environment = dict(os.environ, OPENGLCONTEXT_HIDDEN='1',
                       OPENGLCONTEXT_NO_VSYNC='1')
    completed = subprocess.run(
        [VIEWER, result.tileset, '--capture', capture, '--capture-delay', '4',
         '--frames', '90', '--size', '640x360',
         '--eye', '0,95,60', '--look-at', '60,80,-120'],
        env=environment, capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert os.path.exists(capture), completed.stdout[-2000:]
    return np.asarray(Image.open(capture).convert('RGB'), dtype='d') / 255.0


def _rows(pixels, low, high):
    height = pixels.shape[0]
    return pixels[int(height * low):int(height * high)]


class TestTheFrameThatComesBack:
    def test_something_was_drawn(self, rendered) -> None:
        assert rendered.shape[:2] == (360, 640)
        assert rendered.std() > 0.02, "the frame is a flat colour"

    def test_the_sky_is_above_and_the_ground_below(self, rendered) -> None:
        sky = _rows(rendered, 0.0, 0.15).reshape(-1, 3).mean(axis=0)
        ground = _rows(rendered, 0.7, 1.0).reshape(-1, 3).mean(axis=0)
        assert sky[2] > sky[0], "the top of the frame is not sky-blue"
        assert ground[1] >= ground[2], "the bottom of the frame is not ground"

    def test_the_ground_is_not_a_silhouette(self, rendered) -> None:
        """Lit geometry, not a flat unlit fill: the ground half varies."""
        assert _rows(rendered, 0.5, 1.0).std() > 0.03

    def test_the_trees_are_there(self, rendered) -> None:
        """Conifer green is darker and more saturated than the ground it is on."""
        pixels = _rows(rendered, 0.3, 1.0).reshape(-1, 3)
        green = pixels[:, 1]
        saturation = green - np.maximum(pixels[:, 0], pixels[:, 2])
        assert (saturation > 0.08).mean() > 0.01, "no strongly green pixels"

    def test_the_frame_is_not_mostly_empty(self, rendered) -> None:
        """A world that failed to stream leaves the background showing."""
        background = np.abs(rendered - rendered[0, 0]).sum(axis=2) < 0.02
        assert background.mean() < 0.6
