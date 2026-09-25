"""A world document's figures, read back: a value it gets wrong costs that value.

Each of these reads a piece of a project file a designer saved, or a file
someone else wrote. A figure that is not a number takes its default and is
logged; one out of range is held to it; the project still opens.
"""

from OpenGLContext_editor.world.dem import DEMBase
from OpenGLContext_editor.world.height import DEFAULT_RELIEF, ProceduralBase
from OpenGLContext_editor.world.hydrology import BED_DEPTH, BED_WIDTH, Channel
from OpenGLContext_editor.world.presets import PresetBase
from OpenGLContext_editor.world.sculpt import SculptStroke


def test_a_dem_base_with_a_bad_relief_takes_the_default(caplog):
    base = DEMBase.from_json({'path': 'x.tif', 'relief': 'tall', 'centre': ['a', 1]})
    assert base.relief == 1.0
    assert base.centre == (0.0, 0.0)
    assert 'relief' in caplog.text


def test_a_procedural_base_with_an_infinite_relief_takes_the_default():
    assert ProceduralBase.from_json({'relief': float('inf')}).relief == DEFAULT_RELIEF


def test_a_channel_with_a_negative_width_is_held_to_zero():
    channel = Channel.from_json({'points': [[0, 0], [1, 1]], 'width': -4, 'depth': 'deep'})
    assert (channel.width, channel.depth) == (0.0, BED_DEPTH)
    assert Channel.from_json({}).width == BED_WIDTH


def test_a_preset_with_a_seed_that_is_not_a_number_takes_the_default():
    preset = PresetBase.from_json({'seed': 'lucky', 'relief': 'nan'})
    assert (preset.seed, preset.relief) == (0, 1.0)


def test_a_stroke_with_a_bad_figure_keeps_the_others():
    stroke = SculptStroke.from_json({'radius': 'wide', 'amount': 5, 'seed': 2.5})
    assert (stroke.radius, stroke.amount, stroke.seed) == (100.0, 5.0, 0)
