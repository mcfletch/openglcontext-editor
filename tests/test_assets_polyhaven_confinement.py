"""What a library's answer is allowed to make this do.

A fetch asks Poly Haven where an asset's files are and is told, in JSON: a URL
per file, and the *path to write it to* as the key beside it. Both halves of
that are data. A service that is compromised, misconfigured or simply wrong can
answer with a path that climbs out of the download directory -- which is a write
of bytes it also chose, anywhere the user can write -- or with a URL somewhere
else entirely.

So the path is resolved under the directory being downloaded into, and the URL
is checked against the hosts Poly Haven publishes from before a socket is
opened.
"""

import os

import pytest

from OpenGLContext_editor.assets import polyhaven


def _library(include=None):
    """What the library says it publishes for one asset."""
    include = {'fern_02.bin': {'url': 'https://dl/fern_02.bin'},
               'textures/fern_02_diff_1k.jpg': {'url': 'https://dl/diff.jpg'},
               } if include is None else include
    return {'gltf': {'1k': {'gltf': {'url': 'https://dl/fern_02_1k.gltf',
                                     'include': include}}}}


def _transport(published, payload=b'bytes'):
    def get(url):
        import json
        if '/files/' in url:
            return json.dumps(published).encode()
        if '/info/' in url:
            return json.dumps({'name': 'Fern', 'authors': {'Someone': 'All'}}).encode()
        return payload
    return get


class TestWhereADownloadMayBeWritten:
    def test_an_ordinary_layout_is_written_under_the_directory(self, tmp_path):
        got = polyhaven.fetch('fern_02', str(tmp_path),
                              transport=_transport(_library()))

        assert os.path.realpath(got.diffuse).startswith(os.path.realpath(str(tmp_path)))

    def test_a_path_climbing_out_of_the_directory_is_refused(self, tmp_path):
        published = _library({
            'fern_02.bin': {'url': 'https://dl/fern_02.bin'},
            '../../../.bashrc': {'url': 'https://dl/diff.jpg'},
        })

        with pytest.raises(IOError):
            polyhaven.fetch('fern_02', str(tmp_path / 'downloads'),
                            transport=_transport(published, b'# owned'))

    def test_nothing_was_written_outside(self, tmp_path):
        """The one that matters: the refusal has to happen before the write."""
        outside = tmp_path / 'target.txt'
        outside.write_text('mine')
        published = _library({
            '../target.txt': {'url': 'https://dl/diff.jpg'},
        })

        with pytest.raises(IOError):
            polyhaven.fetch('fern_02', str(tmp_path / 'downloads'),
                            transport=_transport(published, b'# owned'))

        assert outside.read_text() == 'mine'

    def test_an_absolute_path_is_refused(self, tmp_path):
        published = _library({
            str(tmp_path / 'absolute.bin'): {'url': 'https://dl/diff.jpg'},
        })

        with pytest.raises(IOError):
            polyhaven.fetch('fern_02', str(tmp_path / 'downloads'),
                            transport=_transport(published))


class TestWhereBytesMayBeFetchedFrom:
    def test_a_polyhaven_url_is_fetched(self, monkeypatch):
        opened = []
        monkeypatch.setattr(polyhaven, '_open_capped',
                            lambda url, cap: opened.append(url) or b'ok')

        assert polyhaven._get('https://dl.polyhaven.org/file/x.gltf') == b'ok'
        assert opened == ['https://dl.polyhaven.org/file/x.gltf']

    def test_the_library_itself_is_fetched(self, monkeypatch):
        monkeypatch.setattr(polyhaven, '_open_capped', lambda url, cap: b'{}')

        assert polyhaven._get('https://api.polyhaven.com/files/fern_02') == b'{}'

    def test_another_host_is_refused(self, monkeypatch):
        opened = []
        monkeypatch.setattr(polyhaven, '_open_capped',
                            lambda url, cap: opened.append(url) or b'')

        with pytest.raises(IOError):
            polyhaven._get('https://evil.example/payload')

        assert opened == []

    def test_a_file_url_is_refused(self):
        with pytest.raises(IOError):
            polyhaven._get('file:///etc/passwd')


class TestHowMuchMayArrive:
    def test_a_body_is_capped(self, monkeypatch):
        """A download is read into memory before it is written."""
        capped = []
        monkeypatch.setattr(polyhaven, '_open_capped',
                            lambda url, cap: capped.append(cap) or b'')

        polyhaven._get('https://dl.polyhaven.org/file/x.bin')

        assert capped == [polyhaven.MAX_DOWNLOAD_BYTES]
