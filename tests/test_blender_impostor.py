"""Baking an impostor atlas in a real Blender, and checking what landed where.

An impostor is a bargain: the baker puts the view from direction *d* in a
particular tile, and the renderer reads that tile when it is looked at from *d*.
Everything between them is a chance to get a flip wrong -- Blender hands pixels
back bottom row first, an image file is read top row first, and a texture
coordinate's v runs the other way again. A mirrored atlas raises nothing and
crashes nothing; it just shows the back of the model when you are in front of
it.

So the subject is a cube whose six faces *emit* six different colours. Emission
rather than paint, so the pixel is the colour and not the colour under a
lighting model -- what is being tested is where the view landed, not how it was
shaded.

Under an orthographic projection a cube face covers screen area in proportion
to ``n . d``, so the average colour of a tile can be predicted exactly: the
area-weighted mix of the faces pointing at that tile's direction. Measured
against predicted, for all sixteen tiles. Only one arrangement of flips makes
that true.

Blender is a separate application rather than a dependency: these run where one
is on the path and are skipped where none is.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from OpenGLContext_editor import blender
from OpenGLContext_editor.blender.openglcontext_lod import impostorspec, octahedral

pytestmark = pytest.mark.blender

GRID = 4
ATLAS = 128

#: The cube's faces, as (normal, colour). Saturated and far apart so the
#: average of a tile names one of them without ambiguity.
FACES = [
    ((1, 0, 0), (1.0, 0.0, 0.0)),      # +X red
    ((-1, 0, 0), (0.0, 1.0, 0.0)),     # -X green
    ((0, 1, 0), (0.0, 0.0, 1.0)),      # +Y blue
    ((0, -1, 0), (1.0, 1.0, 0.0)),     # -Y yellow
    ((0, 0, 1), (1.0, 0.0, 1.0)),      # +Z magenta
    ((0, 0, -1), (0.0, 1.0, 1.0)),     # -Z cyan
]


@pytest.fixture(scope='module')
def have_blender():
    try:
        return blender.executable()
    except blender.BlenderMissing as error:
        pytest.skip(str(error))


BAKE = '''
import json, sys
sys.path.insert(0, %(addon)r)
import bpy, numpy as np
from openglcontext_lod import impostor, octahedral

GRID, ATLAS = %(grid)d, %(atlas)d
FACES = json.loads(%(faces)r)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add(size=2.0)
cube = bpy.context.active_object
# One material per face, so which face is in shot is a colour.
for normal, colour in FACES:
    material = bpy.data.materials.new('f%%s' %% (normal,))
    material.use_nodes = True
    tree = material.node_tree
    for node in list(tree.nodes):
        if node.type != 'OUTPUT_MATERIAL':
            tree.nodes.remove(node)
    output = next(n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL')
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = tuple(colour) + (1.0,)
    emission.inputs['Strength'].default_value = 1.0
    tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
    cube.data.materials.append(material)
for polygon in cube.data.polygons:
    best, at = -2.0, 0
    for index, (normal, _colour) in enumerate(FACES):
        got = sum(a * b for a, b in zip(polygon.normal, normal))
        if got > best:
            best, at = got, index
    polygon.material_index = at

out = %(out)r
baked = impostor.bake_atlas(cube, out, grid=GRID, image=ATLAS, hemi=True)

# Read the atlas back and report each tile's average colour.
image = bpy.data.images.load(out)
flat = np.zeros(len(image.pixels), dtype='f')
image.pixels.foreach_get(flat)
pixels = flat.reshape((ATLAS, ATLAS, 4))[::-1]      # top row first
tile = ATLAS // GRID
means = []
for row in range(GRID):
    for column in range(GRID):
        patch = pixels[row * tile:(row + 1) * tile,
                       column * tile:(column + 1) * tile]
        solid = patch[patch[:, :, 3] > 0.5]
        means.append([float(v) for v in solid[:, :3].mean(axis=0)]
                     if len(solid) else None)
print('MARK ' + json.dumps({
    'means': means, 'radius': baked.radius,
    'centre': list(baked.centre), 'grid': baked.grid,
    'exists': __import__('os').path.getsize(out) > 0,
}))
'''


@pytest.fixture(scope='module')
def baked(tmp_path_factory, have_blender):
    where = tmp_path_factory.mktemp('impostor')
    script = where / 'bake.py'
    out = str(where / 'atlas.png')
    script.write_text(textwrap.dedent(BAKE) % {
        'addon': blender.addon_directory().rsplit(os.sep, 1)[0],
        'grid': GRID, 'atlas': ATLAS, 'out': out,
        'faces': json.dumps([[list(n), list(c)] for n, c in FACES]),
    })
    done = subprocess.run(
        [have_blender, '--background', '--factory-startup', '-P', str(script)],
        capture_output=True, text=True, timeout=900)
    if 'MARK ' not in done.stdout:
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-4000:])
        raise AssertionError('the bake printed no result')
    said = json.loads(done.stdout.split('MARK ', 1)[1].splitlines()[0])
    said['path'] = out
    return said


def expected_mix(direction):
    """The colour a tile looking from ``direction`` averages to.

    Orthographic, so a face covers screen area in proportion to ``n . d`` and
    the average over the covered pixels is that weighted mix of the faces
    facing the camera.
    """
    weights = [(max(0.0, sum(a * b for a, b in zip(direction, normal, strict=True))), colour)
               for normal, colour in FACES]
    total = sum(weight for weight, _colour in weights)
    return [sum(weight * colour[channel] for weight, colour in weights) / total
            for channel in range(3)]


def dominant_face(direction):
    """Which single face is most nearly facing ``direction``."""
    return max(FACES, key=lambda face: sum(
        a * b for a, b in zip(direction, face[0], strict=True)))[1]


class TestTheAtlasWasWritten:
    def test_there_is_a_file(self, baked):
        assert baked['exists'] and os.path.exists(baked['path'])

    def test_every_tile_has_something_in_it(self, baked):
        assert all(mean is not None for mean in baked['means'])

    def test_the_card_is_as_big_as_the_model(self, baked):
        """A unit cube's bounding sphere, with the bake's margin on it."""
        assert 1.7 < baked['radius'] < 2.0

    def test_the_card_is_centred_on_the_model(self, baked):
        assert baked['centre'] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


class TestEachTileHoldsTheViewItShould:
    """The one arrangement of flips that makes this true is the right one."""

    def test_every_tile_holds_the_view_from_its_own_direction(self, baked):
        directions = octahedral.view_directions(GRID, hemi=True)
        wrong = []
        for index, direction in enumerate(directions):
            measured = baked['means'][index]
            wanted = expected_mix(direction)
            if max(abs(a - b) for a, b in zip(measured, wanted, strict=True)) > 0.12:
                wrong.append((index, [round(v, 2) for v in direction],
                              [round(v, 2) for v in measured],
                              [round(v, 2) for v in wanted]))

        assert not wrong, ('tiles holding the wrong view (index, direction, '
                           'measured, wanted): %r' % (wrong[:4],))

    def test_the_up_facing_view_is_the_top_of_the_cube(self, baked):
        """+Y is up, so the middle of the square looks down on it."""
        directions = octahedral.view_directions(GRID, hemi=True)
        for index, direction in enumerate(directions):
            if direction[1] > 0.95:
                assert baked['means'][index] == pytest.approx(
                    [0.0, 0.0, 1.0], abs=0.1)

    def test_east_and_west_do_not_show_the_same_face(self, baked):
        """The failure a mirrored atlas passes most other tests with."""
        directions = octahedral.view_directions(GRID, hemi=True)
        east = [baked['means'][i] for i, d in enumerate(directions) if d[0] > 0.7]
        west = [baked['means'][i] for i, d in enumerate(directions) if d[0] < -0.7]

        assert east and west
        # +X emits red, -X emits green: the two sides cannot look alike.
        assert min(c[0] for c in east) > max(c[0] for c in west)
        assert min(c[1] for c in west) > max(c[1] for c in east)


class TestWhichEngineRendersTheViews:
    """EEVEE is named BLENDER_EEVEE_NEXT in Blender 4.2 to 4.x and
    BLENDER_EEVEE again from 5.0; the bake takes whichever is offered."""

    def test_blender_4_names_it_next(self):
        offered = ['BLENDER_EEVEE_NEXT', 'BLENDER_WORKBENCH', 'CYCLES']
        assert impostorspec.render_engine(offered) == 'BLENDER_EEVEE_NEXT'

    def test_blender_5_names_it_eevee(self):
        offered = ['BLENDER_EEVEE', 'BLENDER_WORKBENCH', 'CYCLES']
        assert impostorspec.render_engine(offered) == 'BLENDER_EEVEE'

    def test_without_eevee_the_workbench_draws(self):
        assert impostorspec.render_engine(['BLENDER_WORKBENCH', 'CYCLES']) \
            == 'BLENDER_WORKBENCH'
