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
    # which() rather than joining the name on: it applies PATHEXT, and the
    # console script is oglc-view.exe on Windows. Joining finds nothing there,
    # and the search falls through to a PATH that a venv run this way is not on.
    beside = shutil.which('oglc-view', path=os.path.dirname(sys.executable))
    if beside:
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
    # Stand on the circuit and look along it, rather than at a pose written
    # down here: what the world looks like is the world's business, and a fixed
    # camera silently stops meaning anything the moment the terrain changes.
    # Somewhere the road is on the ground, since what is being looked at is the
    # ground and the trees on it -- from inside a bore there is only lining.
    circuit = world.circuit()
    line = circuit.points
    at = _on_the_ground(circuit, ahead=25)
    eye = line[at] + np.array([0.0, 1.6, 0.0])
    aim = line[(at + 25) % len(line)] + np.array([0.0, 1.2, 0.0])
    capture = os.path.join(str(directory), 'view.png')
    environment = dict(os.environ, OPENGLCONTEXT_HIDDEN='1',
                       OPENGLCONTEXT_NO_VSYNC='1')
    completed = subprocess.run(
        [VIEWER, result.tileset, '--capture', capture, '--capture-delay', '4',
         '--frames', '90', '--size', '640x360',
         # Joined with '=': a position west or north of the origin begins with
         # a minus sign, which as a separate argument reads as another option.
         '--eye=%f,%f,%f' % tuple(eye), '--look-at=%f,%f,%f' % tuple(aim)],
        env=environment, capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert os.path.exists(capture), completed.stdout[-2000:]
    return np.asarray(Image.open(capture).convert('RGB'), dtype='d') / 255.0


def _on_the_ground(circuit, ahead):
    """A point on the circuit with ``ahead`` points of open road in front of it.

    The alignment crosses this landscape on bridges and through tunnels for a
    good part of its length, and neither shows the ground.
    """
    laid = circuit.on_ground
    for at in range(len(laid) - ahead):
        if laid[at:at + ahead + 1].all():
            return at
    raise AssertionError("the circuit is on structures from end to end")


def _cast(pixels):
    """How green or how blue a patch of the frame is, past neutral.

    Both of the things that can be overhead have a cast: sky is blue and a
    canopy is green. Tarmac has none, which is what makes this tell which way
    up the frame is without assuming which of the two is up there.

    The strongest tenth of the patch rather than its average, because the two
    casts *cancel*: blue sky between grey-green conifers averages out to a
    neutral the same arithmetic gets from tarmac, and a frame full of both
    then reads as a frame of road. What is being asked is whether anything up
    there is sky or leaf, so what is measured is the pixels that are.
    """
    flat = pixels.reshape(-1, 3)
    green = flat[:, 1] - np.maximum(flat[:, 0], flat[:, 2])
    blue = flat[:, 2] - np.maximum(flat[:, 0], flat[:, 1])
    return float(np.percentile(np.maximum(green, blue), 90))


def _rows(pixels, low, high):
    height = pixels.shape[0]
    return pixels[int(height * low):int(height * high)]


class TestTheFrameThatComesBack:
    def test_something_was_drawn(self, rendered) -> None:
        assert rendered.shape[:2] == (360, 640)
        assert rendered.std() > 0.02, "the frame is a flat colour"

    def test_the_world_is_above_and_the_road_below(self, rendered) -> None:
        """Which way up the frame is.

        Not "the top is sky": a forest road has a canopy over it, and where
        this world is at its best the top of the frame is leaves. What is true
        either way is that the top of the frame is *coloured* -- blue sky or
        green canopy -- and the bottom is the neutral grey of tarmac.

        """
        above = _cast(_rows(rendered, 0.0, 0.08))
        below = _cast(_rows(rendered, 0.85, 1.0))
        assert above > 0.012, "the top of the frame is neither sky nor canopy"
        assert below < 0.012, "the bottom of the frame is not tarmac"

    def test_the_road_is_underfoot(self, rendered) -> None:
        """Standing on the circuit, the bottom of the frame is tarmac: dark,
        and grey rather than green."""
        underfoot = _rows(rendered, 0.9, 1.0).reshape(-1, 3)
        assert underfoot.mean() < 0.35
        greenness = underfoot[:, 1] - np.maximum(underfoot[:, 0], underfoot[:, 2])
        assert abs(float(greenness.mean())) < 0.05

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
