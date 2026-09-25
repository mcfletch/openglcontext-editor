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
                       solidTexture=str(tmp_path / ('%s_bark.png' % name)),
                       foliageTexture=str(tmp_path / ('%s_leaf.png' % name)),
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


def _plants(tmp_path, *names):
    """Several kinds of cover, each with its own files."""
    made = []
    for name in names:
        (tmp_path / ('%s.glb' % name)).write_bytes(b'g')
        (tmp_path / ('%s.png' % name)).write_bytes(b'p')
        made.append(CoverSpecies(name=name,
                                 card=str(tmp_path / ('%s.png' % name)),
                                 clump=str(tmp_path / ('%s.glb' % name)),
                                 density=2.0))
    return made


class TestTheCoverRecord:
    def _grown(self, layer):
        return layer.metadata()['vegetation']['cover']

    def test_a_layer_without_cover_writes_none(self, tmp_path) -> None:
        assert 'cover' not in _layer(tmp_path).metadata()['vegetation']

    def test_a_layer_with_it_writes_the_recipe(self, tmp_path) -> None:
        found = self._grown(_layer(tmp_path, cover=_cover(tmp_path)))
        assert found['species'][0]['name'] == 'grass'
        assert found['species'][0]['density'] == 2.0

    def test_one_plant_may_be_given_on_its_own(self, tmp_path) -> None:
        """A world with one kind of cover should not have to say so twice."""
        found = self._grown(_layer(tmp_path, cover=_cover(tmp_path)))
        assert len(found['species']) == 1

    def test_a_whole_set_of_plants_is_carried(self, tmp_path) -> None:
        """A forest floor is grass and fern and nettle, not one plant."""
        layer = _layer(tmp_path,
                       cover=_plants(tmp_path, 'grass', 'fern', 'nettle'))
        assert [one['name'] for one in self._grown(layer)['species']] \
            == ['grass', 'fern', 'nettle']

    def test_the_files_land_under_the_world(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_plants(tmp_path, 'grass', 'fern'))
        assets = layer.assets()
        for one in self._grown(layer)['species']:
            assert one['card'] in assets
            assert one['clump'] in assets

    def test_it_names_no_path_from_this_machine(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_plants(tmp_path, 'grass', 'fern'))
        for one in self._grown(layer)['species']:
            assert not one['card'].startswith('/')
            assert not one['clump'].startswith('/')

    def test_plants_sharing_a_file_carry_it_once(self, tmp_path) -> None:
        """Two variants of one scan share the model it was baked from."""
        (tmp_path / 'fern.glb').write_bytes(b'g')
        (tmp_path / 'a.png').write_bytes(b'p')
        (tmp_path / 'b.png').write_bytes(b'p')
        shared = [CoverSpecies(name=name, card=str(tmp_path / ('%s.png' % name)),
                               clump=str(tmp_path / 'fern.glb'))
                  for name in ('a', 'b')]
        assets = _layer(tmp_path, cover=shared).assets()
        assert len([name for name in assets if name.endswith('fern.glb')]) == 1

    def test_files_named_alike_in_two_places_are_both_carried(
            self, tmp_path) -> None:
        """Two sets baked into two directories both call their card
        ``card.png``; each species keeps its own."""
        made = []
        for name in ('fern', 'shrub'):
            (tmp_path / name).mkdir()
            (tmp_path / name / 'card.png').write_bytes(name.encode())
            (tmp_path / name / 'clump.glb').write_bytes(name.encode())
            made.append(CoverSpecies(
                name=name, card=str(tmp_path / name / 'card.png'),
                clump=str(tmp_path / name / 'clump.glb')))
        layer = _layer(tmp_path, cover=made)
        assets = layer.assets()
        grown = self._grown(layer)['species']
        assert len({one['card'] for one in grown}) == 2
        assert len({one['clump'] for one in grown}) == 2
        for one in grown:
            assert assets[one['card']] == one['name'].encode()
            assert assets[one['clump']] == one['name'].encode()

    def test_it_says_what_it_grows_on(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path),
                       cover_on=['grass', 'forest_floor'])
        assert self._grown(layer)['on'] == ['grass', 'forest_floor']

    def test_cover_with_no_clump_is_cards_all_the_way(self, tmp_path) -> None:
        (tmp_path / 'blade.png').write_bytes(b'p')
        cards = CoverSpecies(name='grass', card=str(tmp_path / 'blade.png'))
        layer = _layer(tmp_path, cover=cards)
        assert self._grown(layer)['species'][0]['clump'] is None
        assert len(layer.assets()) >= 2

    def test_it_is_plain_json(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_cover(tmp_path))
        assert json.loads(json.dumps(layer.metadata()))

    def test_it_reads_back_as_the_species_it_was(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=_plants(tmp_path, 'grass', 'fern'))
        back = [CoverSpecies.from_json(one)
                for one in self._grown(layer)['species']]
        assert [one.name for one in back] == ['grass', 'fern']
        assert back[0].density == 2.0


class TestTheShippedCover:
    def _cover(self):
        from OpenGLContext_editor.world.species import (
            shipped_cover,
            species_are_available,
        )
        if not species_are_available():
            pytest.skip("the example world's assets are not installed")
        return shipped_cover()

    def test_the_example_world_has_some(self) -> None:
        found = self._cover()
        assert found and all(one.card for one in found)

    def test_it_is_more_than_one_kind_of_plant(self) -> None:
        """A floor of one plant repeated is a lawn with trees on it."""
        assert len({one.name for one in self._cover()}) > 1

    def test_its_files_are_where_it_says(self) -> None:
        import os
        for one in self._cover():
            assert os.path.exists(one.card)
            if one.clump:
                assert os.path.exists(one.clump)

    def test_a_world_can_be_baked_with_it(self, tmp_path) -> None:
        layer = _layer(tmp_path, cover=self._cover())
        grown = layer.metadata()['vegetation']['cover']['species']
        assert len(grown) == len(self._cover())
        assert all(one['card'] in layer.assets() for one in grown)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
