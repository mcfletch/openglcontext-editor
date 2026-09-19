"""What a chain's sidecars may point at.

A ``glb`` naming a level's bytes in a file beside it is a document naming a
path, and a document is data: the name in it may be anything the person who
wrote the file chose. Reading it as given would let a model handed to a user
read whatever that user can read, which is the oldest bug in every format that
follows a reference.

So the same policy the engine's loaders are held to applies here
(:mod:`OpenGLContext.loaders.resolver`): a level's bytes must be under the
directory the glb is in, and how many bytes may be read is bounded. The cases
below are the references that must be refused.
"""

import json
import os
import struct

import numpy as np
import pytest

from OpenGLContext_editor.meshlod import LODAsset, write_chain
from OpenGLContext_editor.meshlod.chain import LODChain, LODLevel

SECRET = b'the private key nobody asked this model for'


def _level(scale=1.0):
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype='f4') * scale
    return LODLevel(
        attributes={'POSITION': points},
        indices=np.array([0, 1, 2], np.uint32),
        error=0.0,
        vertex_map=np.arange(3, dtype=np.uint32),
    )


@pytest.fixture
def chain_at(tmp_path):
    """A two-level chain written into a subdirectory, with a secret above it."""
    (tmp_path / 'secret.bin').write_bytes(SECRET)
    model = tmp_path / 'model'
    model.mkdir()

    def write(uri=None):
        written = write_chain(str(model / 'bust.glb'),
                              LODChain(levels=[_level(1.0), _level(0.5)],
                                       centre=(0.5, 0.5, 0.0), radius=1.0))
        if uri is not None:
            _repoint(written[0], uri)
        return written[0]

    return write


def _repoint(glb, uri):
    """Rewrite the glb's sidecar buffer to name ``uri`` instead."""
    with open(glb, 'rb') as handle:
        raw = handle.read()
    json_length, _kind = struct.unpack('<II', raw[12:20])
    document = json.loads(raw[20:20 + json_length])
    for buffer in document['buffers']:
        if 'uri' in buffer:
            buffer['uri'] = uri
    encoded = json.dumps(document, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * ((4 - len(encoded) % 4) % 4)
    rest = raw[20 + json_length:]
    head = struct.pack('<III', *struct.unpack('<III', raw[:12])[:2],
                       12 + 8 + len(encoded) + len(rest))
    with open(glb, 'wb') as handle:
        handle.write(head)
        handle.write(struct.pack('<II', len(encoded), 0x4E4F534A))
        handle.write(encoded)
        handle.write(rest)


class TestWhereALevelsBytesMayLive:
    def test_a_sidecar_beside_the_glb_is_read(self, chain_at):
        """The ordinary case, which must keep working."""
        asset = LODAsset.open(chain_at())

        attributes, indices = asset.load(0)

        assert len(indices) == 3
        assert attributes['POSITION'].max() == pytest.approx(1.0)

    def test_a_reference_above_the_glb_is_refused(self, chain_at):
        with pytest.raises(IOError):
            LODAsset.open(chain_at('../secret.bin')).load(0)

    def test_a_deeper_traversal_is_refused(self, chain_at):
        with pytest.raises(IOError):
            LODAsset.open(chain_at('../../../../etc/passwd')).load(0)

    def test_an_absolute_path_is_refused(self, chain_at, tmp_path):
        with pytest.raises(IOError):
            LODAsset.open(chain_at(str(tmp_path / 'secret.bin'))).load(0)

    def test_a_url_is_refused(self, chain_at):
        """A chain is a file and its sidecars are files beside it."""
        with pytest.raises(IOError):
            LODAsset.open(chain_at('http://169.254.169.254/latest/meta-data/')).load(0)

    def test_an_escape_dressed_as_a_percent_escape_is_refused(self, chain_at):
        with pytest.raises(IOError):
            LODAsset.open(chain_at('%2e%2e%2fsecret.bin')).load(0)

    def test_what_it_refused_is_not_in_what_it_returned(self, chain_at):
        """The one that would matter: the secret must not come back as data."""
        asset = LODAsset.open(chain_at('../secret.bin'))

        with pytest.raises(IOError):
            attributes, _indices = asset.load(0)
            assert SECRET not in attributes['POSITION'].tobytes()


class TestHowMuchMayBeRead:
    def test_a_level_larger_than_the_cap_is_refused(self, chain_at):
        glb = chain_at()
        asset = LODAsset.open(glb, max_resource_bytes=8)

        with pytest.raises(ValueError):
            asset.load(0)

    def test_the_cap_does_not_stop_an_ordinary_level(self, chain_at):
        asset = LODAsset.open(chain_at(), max_resource_bytes=64 * 1024)

        assert len(asset.load(0)[1]) == 3

    def test_a_level_inside_the_glb_is_capped_too(self, chain_at):
        """The embedded chunk is as much a declared length as a sidecar is."""
        asset = LODAsset.open(chain_at(), max_resource_bytes=8)

        with pytest.raises(ValueError):
            asset.load(len(asset.levels) - 1)


class TestABufferThatCarriesItsOwnBytes:
    def test_a_data_uri_level_is_read(self, chain_at):
        """A file from elsewhere may inline a level rather than name a file."""
        glb = chain_at()
        with open(os.path.join(os.path.dirname(glb), 'bust.lod0.bin'), 'rb') as handle:
            payload = handle.read()
        import base64
        _repoint(glb, 'data:application/octet-stream;base64,'
                 + base64.b64encode(payload).decode('ascii'))

        attributes, indices = LODAsset.open(glb).load(0)

        assert len(indices) == 3
        assert attributes['POSITION'].max() == pytest.approx(1.0)

    def test_a_data_uri_is_capped(self, chain_at):
        glb = chain_at()
        with open(os.path.join(os.path.dirname(glb), 'bust.lod0.bin'), 'rb') as handle:
            payload = handle.read()
        import base64
        _repoint(glb, 'data:application/octet-stream;base64,'
                 + base64.b64encode(payload).decode('ascii'))

        with pytest.raises(ValueError):
            LODAsset.open(glb, max_resource_bytes=8).load(0)
