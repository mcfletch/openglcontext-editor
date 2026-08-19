"""The landscapes a designer can start from.

Each preset is a claim about what the ground does -- a canyon has a trough, a
lake country has ground under the waterline, hills stay gentle -- so that is
what is asserted, over a real sample of each.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.height import HeightSource, base_from_json
from OpenGLContext_editor.world.presets import PRESETS, PresetBase

EXTENT = 4096.0


def _ground(base, steps=97):
    axis = np.linspace(-EXTENT / 2.0, EXTENT / 2.0, steps)
    x, z = np.meshgrid(axis, axis, indexing='ij')
    return np.asarray(base.sample(x, z))


def _preset(name, **named):
    return PresetBase(name=name, **named)


class TestWhatIsOnOffer:
    def test_the_dramatic_four_are_there(self) -> None:
        assert {'mountains', 'lakes', 'hills', 'canyon'} <= set(PRESETS)

    def test_each_has_a_sentence_saying_what_it_is(self) -> None:
        assert all(preset.description for preset in PRESETS.values())

    def test_each_has_a_name_to_put_in_a_menu(self) -> None:
        assert all(preset.label for preset in PRESETS.values())

    def test_a_preset_nobody_ships_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _preset('atlantis').sample(np.zeros(1), np.zeros(1))


class TestWhatEachOneIs:
    def test_mountains_are_the_tallest_of_them(self) -> None:
        relief = {name: float(np.ptp(_ground(_preset(name)))) for name in PRESETS}
        assert relief['mountains'] == max(relief.values())

    def test_hills_are_country_a_road_can_climb(self) -> None:
        """Gentle: nothing on it approaches what a mountain preset does."""
        assert np.ptp(_ground(_preset('hills'))) < 120.0

    def test_a_canyon_has_a_deep_trough_through_it(self) -> None:
        ground = _ground(_preset('canyon'))
        # The channel runs roughly down the middle in x, wandering with z.
        assert ground.min() < np.median(ground) - 150.0

    def test_lake_country_has_ground_under_the_waterline(self) -> None:
        from OpenGLContext.loaders.tiles3d.procedural import WATER_LEVEL
        ground = _ground(_preset('lakes'))
        assert (ground < WATER_LEVEL).mean() > 0.10

    def test_hills_have_almost_no_water(self) -> None:
        from OpenGLContext.loaders.tiles3d.procedural import WATER_LEVEL
        assert (_ground(_preset('hills')) < WATER_LEVEL).mean() < 0.05


class TestUsingOne:
    def test_relief_scales_the_whole_landscape(self) -> None:
        full = _ground(_preset('mountains', relief=1.0))
        half = _ground(_preset('mountains', relief=0.5))
        assert np.allclose(half, full * 0.5)

    def test_a_seed_gives_a_different_landscape_of_the_same_kind(self) -> None:
        one = _ground(_preset('mountains', seed=1))
        other = _ground(_preset('mountains', seed=2))
        assert not np.allclose(one, other)
        assert abs(np.ptp(one) - np.ptp(other)) < np.ptp(one) * 0.5

    def test_it_is_a_height_source_like_any_other(self) -> None:
        source = HeightSource(base=_preset('canyon'))
        axis = np.linspace(-100.0, 100.0, 5)
        x, z = np.meshgrid(axis, axis, indexing='ij')
        assert source.height_fn()(x, z).shape == x.shape

    def test_it_round_trips_through_a_project_file(self) -> None:
        base = _preset('lakes', relief=0.75, seed=6)
        again = base_from_json(base.to_json())
        assert again == base


class TestBeingFoundByName:
    """A project file names a kind; nothing in it says which module to import."""

    def test_a_preset_is_known_without_importing_its_module(self) -> None:
        import subprocess
        import sys
        code = (
            "from OpenGLContext_editor.world.height import base_from_json;"
            "base = base_from_json({'kind': 'preset', 'name': 'canyon'});"
            "print(type(base).__name__)"
        )
        done = subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == 'PresetBase'
