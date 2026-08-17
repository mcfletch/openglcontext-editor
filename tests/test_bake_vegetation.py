"""A forest, and what grows on the ground between its trees, written once.

Where the trees stand is decided at bake time and travels as a table. What
covers the ground is *not* decided at bake time -- the ground is the same
everywhere and there is far too much of it -- so what travels is the recipe: a
clump, a card, how dense, and which of the splat map's layers it grows on.
"""
import json

import numpy as np
import pytest
from OpenGLContext.scenegraph.vegetation.cover import CoverSpecies
from OpenGLContext.scenegraph.vegetation.field import TreeSpecies

from OpenGLContext_editor.bake.vegetation import VegetationLayer


def _species(tmp_path, name='fir'):
    for part in ('%s.npz' % name, '%s_bark.png' % name,
                 '%s_leaf.png' % name, '%s_imp.png' % name):
        (tmp_path / part).write_bytes(b'x')
    return TreeSpecies(name=name, mesh=str(tmp_path / ('%s.npz' % name)),
                       solid_texture=str(tmp_path / ('%s_bark.png' % name)),
                       foliage_texture=str(tmp_path / ('%s_leaf.png' % name)),
                       impostor=str(tmp_path / ('%s_imp.png' % name)))


def _cover(tmp_path):
    (tmp_path / 'clump.glb').write_bytes(b'g')
    (tmp_path / 'blade.png').write_bytes(b'p')
    return CoverSpecies(name='grass', card=str(tmp_path / 'blade.png'),
                        clump=str(tmp_path / 'clump.glb'), density=2.0)


def _layer(tmp_path, **named):
    named.setdefault('positions', np.zeros((12, 3)))
    named.setdefault('heights', np.full(12, 14.0))
    named.setdefault('species', [_species(tmp_path)])
    return VegetationLayer(**named)


class TestTheCoverRecord:
    def test_a_layer_without_cover_writes_none(self, tmp_path) -> None:
        assert 'cover' not in _layer(tmp_path).metadata()['vegetation']

    def test_a_layer_with_it_writes_the_recipe(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        found = layer.metadata()['vegetation']['cover']
        assert found['name'] == 'grass'
        assert found['density'] == 2.0

    def test_the_files_land_under_the_world(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        found = layer.metadata()['vegetation']['cover']
        assets = layer.assets()
        assert found['card'] in assets
        assert found['clump'] in assets

    def test_it_names_no_path_from_this_machine(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        found = layer.metadata()['vegetation']['cover']
        assert not found['card'].startswith('/')
        assert not found['clump'].startswith('/')

    def test_it_says_what_it_grows_on(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path),
                       cover_on=['grass', 'forest_floor'])
        assert layer.metadata()['vegetation']['cover']['on'] \
            == ['grass', 'forest_floor']

    def test_cover_with_no_clump_is_cards_all_the_way(self, tmp_path) -> None:
        (tmp_path / 'blade.png').write_bytes(b'p')
        cards = CoverSpecies(name='grass', card=str(tmp_path / 'blade.png'))
        layer = _layer(tmp_path, cover=cards)
        assert layer.metadata()['vegetation']['cover']['clump'] is None
        assert len(layer.assets()) >= 2

    def test_it_is_plain_json(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        assert json.loads(json.dumps(layer.metadata()))

    def test_it_reads_back_as_the_species_it_was(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        found = layer.metadata()['vegetation']['cover']
        back = CoverSpecies.from_json(found)
        assert back.name == 'grass' and back.density == 2.0


class TestTheShippedCover:
    def test_the_example_world_has_some(self) -> None:
        from OpenGLContext_editor.world.species import (
            shipped_cover,
            species_are_available,
        )
        if not species_are_available():
            pytest.skip("the example world's assets are not installed")
        found = shipped_cover()
        assert found.card and found.clump

    def test_its_files_are_where_it_says(self) -> None:
        import os

        from OpenGLContext_editor.world.species import (
            shipped_cover,
            species_are_available,
        )
        if not species_are_available():
            pytest.skip("the example world's assets are not installed")
        found = shipped_cover()
        assert os.path.exists(found.card)
        assert os.path.exists(found.clump)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
