"""``oglc-bake-plants``: what the command writes for a world to grow.

The one behaviour worth holding here is that a set is built up rather than
replaced. Plants do not all want the same settings -- a shrub of a hundred and
fifty thousand triangles needs a budget a grass tuft does not -- so a set is
baked in several passes into one directory, and a pass that threw away what the
last one wrote would make that impossible.
"""
import json

import pytest
from OpenGLContext.scenegraph.vegetation.cover import CoverSpecies

from OpenGLContext_editor.bin import plants as command


def _entry(name, **named):
    named.setdefault('card', '%s_card.png' % (name,))
    return CoverSpecies(name=name, **named)


class TestTheSetItWrites:
    def test_it_names_what_was_baked(self, tmp_path) -> None:
        command.write_cover(str(tmp_path), [_entry('fern_a'), _entry('grass_a')])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert [one['name'] for one in written['species']] \
            == ['fern_a', 'grass_a']

    def test_a_second_pass_adds_to_the_first(self, tmp_path) -> None:
        command.write_cover(str(tmp_path), [_entry('fern_a')])
        command.write_cover(str(tmp_path), [_entry('shrub_a')])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert [one['name'] for one in written['species']] \
            == ['fern_a', 'shrub_a']

    def test_baking_a_plant_again_replaces_what_it_had(self, tmp_path) -> None:
        command.write_cover(str(tmp_path), [_entry('fern_a', density=1.0)])
        command.write_cover(str(tmp_path), [_entry('fern_a', density=0.25)])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert len(written['species']) == 1
        assert written['species'][0]['density'] == 0.25

    def test_a_plant_of_another_asset_by_the_same_name_is_refused(
            self, tmp_path) -> None:
        """Two scans may both call a node ``Plant``; neither may take the
        other's entry."""
        command.write_cover(str(tmp_path), [_entry('plant', clump='fern_02.glb')])
        with pytest.raises(ValueError, match='fern_02.glb'):
            command.write_cover(str(tmp_path),
                           [_entry('plant', clump='shrub_04.glb')])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert written['species'][0]['clump'] == 'fern_02.glb'

    def test_two_assets_in_one_run_by_the_same_name_are_refused(
            self, tmp_path) -> None:
        with pytest.raises(ValueError):
            command.write_cover(str(tmp_path),
                           [_entry('plant', clump='fern_02.glb'),
                            _entry('plant', clump='shrub_04.glb')])
        assert not (tmp_path / 'cover.json').exists()

    def test_what_it_writes_reads_back_as_species(self, tmp_path) -> None:
        command.write_cover(str(tmp_path), [_entry('fern_a', density=0.3)])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert CoverSpecies.from_json(written['species'][0]).density == 0.3


class TestHowItIsAsked:
    def test_the_canopy_band_is_tree_closure(self) -> None:
        """The band is how much tree cover a plant grows under, 0 on open
        ground, as the engine reads it; not how much light reaches it."""
        # argparse has no public lookup of one option's action.
        found = command.build_arg_parser()._option_string_actions['--canopy']  # noqa: SLF001 argparse
        assert found.metavar == ('LEAST', 'MOST')
        assert 'open ground' in found.help
        assert 'light' not in found.help

    def test_a_plain_name_takes_the_default_density(self) -> None:
        options = command.build_arg_parser().parse_args(['fern_02'])
        assert options.assets == [('fern_02', None)]

    def test_a_density_may_be_given_with_the_name(self) -> None:
        options = command.build_arg_parser().parse_args(['fern_02=0.3'])
        assert options.assets == [('fern_02', 0.3)]

    def test_something_that_is_not_a_density_says_so(self, capsys) -> None:
        with pytest.raises(SystemExit):
            command.build_arg_parser().parse_args(['fern_02=thick'])
        assert 'a density is plants per square metre' in capsys.readouterr().err


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
