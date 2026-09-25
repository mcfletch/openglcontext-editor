"""Fetching a published plant, without touching the network.

The transport is injectable, so these drive the real logic -- which file of the
several published is the cutout mask, what a re-fetch costs, what an asset with
no geometry does -- against a recorded shape of the library's answers.
"""
import json

import pytest

from OpenGLContext_editor.assets import polyhaven


def _library(mask='Alpha', formats=('png', 'jpg'), gltf=True):
    """The shape the library answers `/files/<slug>` in."""
    published = {}
    if gltf:
        published['gltf'] = {'1k': {'gltf': {
            'url': 'https://dl/fern_02_1k.gltf',
            'include': {
                'fern_02.bin': {'url': 'https://dl/fern_02.bin'},
                'textures/fern_02_diff_1k.jpg': {'url': 'https://dl/diff.jpg'},
                'textures/fern_02_nor_gl_1k.jpg': {'url': 'https://dl/nor.jpg'},
            }}}}
    if mask:
        published[mask] = {'1k': {kind: {'url': 'https://dl/mask.%s' % (kind,)}
                                  for kind in formats}}
    return published


def _transport(published, seen=None):
    """Answers the two API calls and hands back a byte for anything else."""
    described = {'name': 'Fern 02',
                 'authors': {'Rob Tuytel': 'scanning',
                             'Rico Cilliers': 'modeling'},
                 'polycount': 6232}

    def get(url):
        if seen is not None:
            seen.append(url)
        if '/files/' in url:
            return json.dumps(published).encode()
        if '/info/' in url:
            return json.dumps(described).encode()
        return b'the bytes of ' + url.encode()
    return get


class TestBringingTheMaskAsWell:
    """The glTF's base colour is a JPEG and cannot hold an alpha channel, so a
    cutout plant fetched without its mask is a fan of opaque rectangles."""

    def test_the_mask_comes_too(self, tmp_path) -> None:
        found = polyhaven.fetch('fern_02', str(tmp_path),
                                transport=_transport(_library()))
        assert found.mask is not None
        assert open(found.mask, 'rb').read().startswith(b'the bytes of')

    def test_the_lossless_mask_is_preferred(self, tmp_path) -> None:
        """A mask is a hard edge, and JPEG ringing along one shows as a fringe
        of half-transparent pixels round every leaf."""
        seen = []
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library(), seen))
        assert 'https://dl/mask.png' in seen
        assert 'https://dl/mask.jpg' not in seen

    def test_jpg_will_do_where_there_is_no_png(self, tmp_path) -> None:
        seen = []
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library(formats=('jpg',)), seen))
        assert 'https://dl/mask.jpg' in seen

    @pytest.mark.parametrize('key', ['Alpha', 'opacity', 'Mask'])
    def test_whatever_that_asset_happens_to_call_it(self, tmp_path, key) -> None:
        """The name is the publisher's choice per asset, not a convention."""
        found = polyhaven.fetch('fern_02', str(tmp_path),
                                transport=_transport(_library(mask=key)))
        assert found.mask is not None

    def test_solid_geometry_simply_has_none(self, tmp_path) -> None:
        """A trunk is opaque all over; there is nothing to cut out."""
        found = polyhaven.fetch('fern_02', str(tmp_path),
                                transport=_transport(_library(mask=None)))
        assert found.mask is None


class TestWhatItPutsOnDisk:
    def test_the_model_and_its_buffer_and_its_colour(self, tmp_path) -> None:
        found = polyhaven.fetch('fern_02', str(tmp_path),
                                transport=_transport(_library()))
        assert found.gltf.endswith('fern_02_1k.gltf')
        assert found.diffuse.endswith('fern_02_diff_1k.jpg')
        assert (tmp_path / 'fern_02' / 'fern_02.bin').exists()

    def test_a_second_fetch_asks_for_nothing(self, tmp_path) -> None:
        """A re-bake should not re-download eight models."""
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library()))
        seen = []
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library(), seen))
        assert not [url for url in seen if 'dl/' in url]

    def test_it_becomes_what_the_bake_wants(self, tmp_path) -> None:
        source = polyhaven.fetch('fern_02', str(tmp_path),
                                 transport=_transport(_library())).source()
        assert source.slug == 'fern_02'
        assert source.credit


class TestWhatItRefuses:
    def test_an_asset_with_no_geometry(self, tmp_path) -> None:
        with pytest.raises(LookupError):
            polyhaven.fetch('grass_floor', str(tmp_path),
                            transport=_transport(_library(gltf=False)))

    def test_a_resolution_it_is_not_published_at(self, tmp_path) -> None:
        with pytest.raises(LookupError) as raised:
            polyhaven.fetch('fern_02', str(tmp_path), resolution='16k',
                            transport=_transport(_library()))
        assert '1k' in str(raised.value)          # says what there is


class TestNotAskingTwiceForWhatIsAlreadyHere:
    """Poly Haven gives this work away. Re-fetching bytes that are on the disk
    spends their bandwidth for nothing."""

    def test_the_librarys_answers_are_kept(self, tmp_path) -> None:
        seen = []
        get = _transport(_library(), seen)
        polyhaven.files('fern_02', get, str(tmp_path))
        polyhaven.files('fern_02', get, str(tmp_path))
        assert len([url for url in seen if '/files/' in url]) == 1

    def test_a_whole_second_fetch_asks_nothing_at_all(self, tmp_path) -> None:
        """Including the two API calls, not just the downloads."""
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library()))
        seen = []
        polyhaven.fetch('fern_02', str(tmp_path),
                        transport=_transport(_library(), seen))
        assert seen == []

    def test_a_second_world_wanting_the_same_plant_asks_nothing(
            self, tmp_path, monkeypatch) -> None:
        """The cache is per user, not per output directory, so baking the same
        fern into two worlds downloads it once."""
        monkeypatch.setenv(polyhaven.CACHE_VARIABLE, str(tmp_path / 'shared'))
        polyhaven.fetch('fern_02', transport=_transport(_library()))
        seen = []
        polyhaven.fetch('fern_02', transport=_transport(_library(), seen))
        assert seen == []

    def test_it_caches_where_it_is_told_to(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(polyhaven.CACHE_VARIABLE, str(tmp_path / 'named'))
        assert polyhaven.cache_dir() == str(tmp_path / 'named')

    def test_a_cache_the_user_cannot_be_surprised_by(self, tmp_path,
                                                     monkeypatch) -> None:
        """Beside the rest of OpenGLContext's downloads, not in shared temp
        where another account could pre-seed a file this one then bakes."""
        monkeypatch.delenv(polyhaven.CACHE_VARIABLE, raising=False)
        monkeypatch.setattr('OpenGLContext.userpaths.appdatadirectory',
                            lambda: str(tmp_path))
        assert polyhaven.cache_dir() == str(
            tmp_path / 'OpenGLContext' / 'polyhaven')


class TestAnInterruptedWrite:
    """A cached file is trusted by being there, so a write that stops part way
    must leave nothing at the name a later run reads."""

    @pytest.fixture
    def interrupted(self, monkeypatch):
        """The process stops after writing and before the file is in place."""
        def stop(_source, _target):
            raise KeyboardInterrupt
        monkeypatch.setattr('os.replace', stop)

    @pytest.mark.usefixtures('interrupted')
    def test_a_download_is_not_left_part_written(self, tmp_path) -> None:
        with pytest.raises(KeyboardInterrupt):
            polyhaven._save('https://dl/x.bin', str(tmp_path / 'x.bin'),  # noqa: SLF001 white-box test of the helper
                            lambda _url: b'the whole file')
        assert not (tmp_path / 'x.bin').exists()

    @pytest.mark.usefixtures('interrupted')
    def test_an_answer_is_not_left_part_written(self, tmp_path) -> None:
        with pytest.raises(KeyboardInterrupt):
            polyhaven.files('fern_02', _transport(_library()), str(tmp_path))
        assert not (tmp_path / '_api' / 'files-fern_02.json').exists()


class TestWhatAnAssetMayBeCalled:
    """A slug names a file and a directory in the cache."""

    @pytest.mark.parametrize('slug', ['../outside', 'a/b', '/etc/passwd', '',
                                      '..', 'fern 02'])
    def test_a_name_that_is_not_a_slug_is_refused(self, tmp_path, slug) -> None:
        seen = []
        with pytest.raises(ValueError):
            polyhaven.fetch(slug, str(tmp_path),
                            transport=_transport(_library(), seen))
        assert seen == []
        assert not (tmp_path / 'outside').exists()

    @pytest.mark.parametrize('slug', ['fern_02', 'Periwinkle-1', 'shrub'])
    def test_a_published_slug_is_taken(self, tmp_path, slug) -> None:
        assert polyhaven.fetch(slug, str(tmp_path),
                               transport=_transport(_library())).slug == slug


class TestTheCreditItWrites:
    def test_it_names_the_work_and_everyone_who_made_it(self) -> None:
        line = polyhaven.credit('fern_02', transport=_transport(_library()))
        assert 'Fern 02' in line
        assert 'Rob Tuytel and Rico Cilliers' in line
        assert 'CC0' in line

    def test_it_points_at_where_the_asset_came_from(self) -> None:
        line = polyhaven.credit('fern_02', transport=_transport(_library()))
        assert 'https://polyhaven.com/a/fern_02' in line

    def test_one_author_is_not_listed_as_two(self) -> None:
        line = polyhaven.credit('x', {'name': 'X', 'authors': {'Someone': 'all'}})
        assert 'by Someone,' in line

    def test_an_asset_with_no_named_author_is_still_credited(self) -> None:
        assert 'Poly Haven' in polyhaven.credit('x', {'name': 'X'})


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
