"""The billboard a plant becomes, rendered from the plant itself.

The card and the geometry cross-fade into each other over a distance window, so
a card that disagrees with the geometry about the plant's colour or its outline
shows as a shimmer at exactly the distance the eye is on it. Rendering the
geometry is what makes them agree, and these check that what comes out is a
picture of the plant rather than a blank or a full frame.

Needs a GL context. The container has one; see the engine's CLAUDE.md.
"""
import numpy as np
import pytest
from PIL import Image

from OpenGLContext_editor.assets import plants
from OpenGLContext_editor.assets.card import bake_card, card_coverage

pytest.importorskip('glfw')


@pytest.fixture
def plant(tmp_path):
    """A baked plant file to render a card from: a leaf on a cutout texture."""
    from tests.test_assets_plants import _source
    source = _source(tmp_path, nodes=[('leaf', (0, 0, 0))])
    out = tmp_path / 'assets'
    species = plants.bake(source, str(out), card=False)
    return str(out / species[0].clump), species[0], out


class TestWhatTheCardComesOutAs:
    def test_it_writes_a_picture_with_an_alpha_channel(self, plant) -> None:
        model, species, out = plant
        target = str(out / 'leaf_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        assert Image.open(target).mode == 'RGBA'

    def test_the_transparent_part_is_the_silhouette(self, plant) -> None:
        """What no triangle covered is what the card has to leave out."""
        model, species, out = plant
        target = str(out / 'leaf_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        covered = card_coverage(target)
        assert 0.02 < covered < 0.95

    def test_the_plant_is_drawn_not_just_cleared(self, plant) -> None:
        model, species, out = plant
        target = str(out / 'leaf_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        picture = np.asarray(Image.open(target))
        lit = picture[picture[..., 3] > 10]
        assert len(lit)
        assert int(lit[..., :3].max()) > 10          # it has colour in it

    def test_it_says_how_wide_the_plant_was(self, plant) -> None:
        """The billboard is one unit tall and this many across."""
        model, species, _out = plant
        width = bake_card(model, species.clumpMesh, None,
                          str(_out / 'leaf_card.png'), size=64)
        assert width == pytest.approx(1.0, abs=0.2)   # the fixture leaf is square

    def test_a_square_plant_gets_a_square_card(self, plant) -> None:
        model, species, out = plant
        target = str(out / 'leaf_card.png')
        bake_card(model, species.clumpMesh, None, target, size=32)
        assert Image.open(target).size == (32, 32)


class TestACardIsShapedLikeThePlant:
    """A plant two and a half times as wide as it is tall, squeezed into a
    square texture, spends half its horizontal resolution on the squeeze and
    most of its vertical on empty sky. The texture is sized to the plant
    instead; the quad is what stretches it back."""

    def _wide(self, tmp_path, across):
        """A plant ``across`` times wider than it is tall."""
        from tests.test_assets_plants import _source
        tmp_path.mkdir(parents=True, exist_ok=True)
        source = _source(tmp_path,
                         nodes=[('mat', (0, 0, 0), (across, 1.0, 1.0))])
        out = tmp_path / 'assets'
        species = plants.bake(source, str(out), card=False)
        return str(out / species[0].clump), species[0], out

    def test_a_wide_plant_gets_a_wide_card(self, tmp_path) -> None:
        model, species, out = self._wide(tmp_path, 4.0)
        target = str(out / 'mat_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        width, height = Image.open(target).size
        assert height == 64
        assert width > 2 * height

    def test_the_stretch_is_not_unbounded(self, tmp_path) -> None:
        """A hedge strip is fifteen times wider than tall, and a texture that
        shape is mostly a waste of memory."""
        model, species, out = self._wide(tmp_path, 20.0)
        target = str(out / 'mat_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        width, height = Image.open(target).size
        assert width <= 4 * height

    def test_a_wide_plant_wastes_no_more_of_its_card_than_a_square_one(
            self, tmp_path) -> None:
        """Which is the point. Squeezed into a square, a plant four times wider
        than tall occupies a quarter of the height and the rest is sky."""
        model, species, out = self._wide(tmp_path / 'square', 1.0)
        bake_card(model, species.clumpMesh, None, str(out / 'card.png'),
                  size=64)
        square = card_coverage(str(out / 'card.png'))
        model, species, out = self._wide(tmp_path / 'wide', 4.0)
        bake_card(model, species.clumpMesh, None, str(out / 'card.png'),
                  size=64)
        assert card_coverage(str(out / 'card.png')) == pytest.approx(
            square, abs=0.05)


class TestAPlantDeeperThanItIsTall:
    """A spreading ground cover is wider in plan than it is tall. Seen from
    the front, what stands in front of and behind the middle is still the
    plant, and the card has to carry it."""

    def _deep(self, tmp_path, depth):
        """Two unit cards, one ``depth`` in front of the middle, one behind."""
        from tests.test_assets_plants import _source
        tmp_path.mkdir(parents=True, exist_ok=True)
        source = _source(tmp_path, nodes=[('leaf', (0, 0, 0))])
        leaf = plants.flatten(source.gltf)[0]
        count = len(leaf.positions)
        both = plants.Variant(
            name='spread',
            positions=np.concatenate([leaf.positions + (0, 0, depth),
                                      leaf.positions - (0, 0, depth)]),
            normals=np.concatenate([leaf.normals, leaf.normals]),
            uvs=np.concatenate([leaf.uvs, leaf.uvs]),
            indices=np.concatenate([leaf.indices, leaf.indices + count]),
            height=1.0)
        out = tmp_path / 'assets'
        species = plants.bake(source, str(out), variants=[both], card=False)
        return str(out / species[0].clump), species[0], out

    def test_what_stands_in_front_and_behind_is_drawn(self, tmp_path) -> None:
        model, species, out = self._deep(tmp_path, 2.0)
        target = str(out / 'spread_card.png')
        bake_card(model, species.clumpMesh, None, target, size=64)
        assert card_coverage(target) > 0.02

    def test_depth_does_not_change_what_is_drawn(self, tmp_path) -> None:
        """The front view of two cards is the same picture however far apart
        they stand; only the card's width, which spans the plant's reach in
        plan, grows with the depth."""
        model, species, out = self._deep(tmp_path / 'near', 0.2)
        near_width = bake_card(model, species.clumpMesh, None,
                               str(out / 'card.png'), size=64)
        near = card_coverage(str(out / 'card.png')) * near_width
        model, species, out = self._deep(tmp_path / 'far', 2.0)
        far_width = bake_card(model, species.clumpMesh, None,
                              str(out / 'card.png'), size=64)
        far = card_coverage(str(out / 'card.png')) * far_width
        assert far_width > near_width
        assert far == pytest.approx(near, rel=0.15)


class TestATargetThatCannotBeDrawnInto:
    def test_it_is_refused_before_anything_is_drawn(self, plant) -> None:
        model, species, out = plant
        target = out / 'empty_card.png'
        with pytest.raises(RuntimeError, match='framebuffer'):
            bake_card(model, species.clumpMesh, None, str(target), size=0)
        assert not target.exists()


class TestABakedPlantGetsItsCardMeasured:
    def test_the_species_carries_the_width_that_was_rendered(self,
                                                             tmp_path) -> None:
        """Not a guess: the quad has to match what the geometry measured."""
        from tests.test_assets_plants import _source
        source = _source(tmp_path, nodes=[('leaf', (0, 0, 0))])
        out = tmp_path / 'assets'
        species = plants.bake(source, str(out), card=True, card_size=64)
        assert species[0].cardWidth == pytest.approx(1.0, abs=0.2)
        assert (out / species[0].card).exists()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
