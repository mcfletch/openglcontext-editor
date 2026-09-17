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
        command._write(str(tmp_path), [_entry('fern_a'), _entry('grass_a')])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert [one['name'] for one in written['species']] \
            == ['fern_a', 'grass_a']

    def test_a_second_pass_adds_to_the_first(self, tmp_path) -> None:
        command._write(str(tmp_path), [_entry('fern_a')])
        command._write(str(tmp_path), [_entry('shrub_a')])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert [one['name'] for one in written['species']] \
            == ['fern_a', 'shrub_a']

    def test_baking_a_plant_again_replaces_what_it_had(self, tmp_path) -> None:
        command._write(str(tmp_path), [_entry('fern_a', density=1.0)])
        command._write(str(tmp_path), [_entry('fern_a', density=0.25)])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert len(written['species']) == 1
        assert written['species'][0]['density'] == 0.25

    def test_what_it_writes_reads_back_as_species(self, tmp_path) -> None:
        command._write(str(tmp_path), [_entry('fern_a', density=0.3)])
        written = json.loads((tmp_path / 'cover.json').read_text())
        assert CoverSpecies.from_json(written['species'][0]).density == 0.3


class TestHowItIsAsked:
    def test_a_plain_name_takes_the_default_density(self) -> None:
        assert command._wanted('fern_02') == ('fern_02', None)

    def test_a_density_may_be_given_with_the_name(self) -> None:
        assert command._wanted('fern_02=0.3') == ('fern_02', 0.3)

    def test_something_that_is_not_a_density_says_so(self) -> None:
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            command._wanted('fern_02=thick')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
