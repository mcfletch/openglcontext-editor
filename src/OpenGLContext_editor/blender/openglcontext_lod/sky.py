"""The sky a world is roofed with: one panorama, written into the glB.

A hall open to the weather needs three things that are all the same thing: a
backdrop to see between the beams, an environment for the polished floor and
the marble to reflect, and the light that is not the sun. An equirectangular
panorama in ``OMI_environment_sky`` is all three at once -- OpenGLContext reads
it as a skybox *and* registers it as the image-based lighting probe
(``loaders/gltf/environment_sky.py``).

The panorama is generated rather than fetched. Below the horizon it is the
room's own floor colour, so what the parquet reflects is parquet rather than
somebody else's field; above it, a blue gradient with fair-weather cloud drawn
around a named sun azimuth, so the bright side of the sky and the direction the
shadows fall agree. Nothing here imports Blender, and nothing it makes needs a
licence: the sky is arithmetic.

    from . import sky
    blob = sky.with_sky(open('world.glb', 'rb').read(), sky.panorama())

Two pieces: :func:`panorama` makes the picture, :func:`with_sky` puts it in a
document. They are apart because the first is worth looking at on its own.
"""

from __future__ import annotations

import io
import json
import math
import struct
from typing import Any

import numpy as np

__all__ = [
    'EXTENSION', 'GROUND_COLOUR', 'HORIZON_COLOUR', 'ZENITH_COLOUR',
    'panorama', 'with_sky',
]

#: What the document declares the sky under.
EXTENSION = 'OMI_environment_sky'

#: Below the horizon: the polished parquet of the hall, averaged. It is what a
#: reflective floor sees when it looks down, and a sky that puts anything else
#: there reflects a world the room is not in.
GROUND_COLOUR = (0.24, 0.13, 0.07)

#: At the horizon: the haze a distance fades into.
HORIZON_COLOUR = (0.82, 0.86, 0.92)

#: Straight up.
ZENITH_COLOUR = (0.22, 0.42, 0.78)

#: What the sun's own patch of sky is worth, and how wide it is in radians.
SUN_GLOW = 0.55
SUN_WIDTH = 0.45

#: The GLB container: magic, the JSON chunk's type, and the BIN chunk's.
_MAGIC = 0x46546C67
_BIN_CHUNK = 0x004E4942
_JSON_CHUNK = 0x4E4F534A


def _lattice(rows: int, columns: int, generator: np.random.Generator
             ) -> np.ndarray:
    """Random values on a grid that wraps in longitude."""
    values = generator.random((rows + 1, columns))
    return np.concatenate([values, values[:, :1]], axis=1)


def _smooth(t: np.ndarray) -> np.ndarray:
    """The usual quintic ease, so the octaves have no visible lattice."""
    return np.asarray(t * t * t * (t * (t * 6.0 - 15.0) + 10.0))


def _noise(shape: tuple[int, int], cells: tuple[int, int],
           generator: np.random.Generator) -> np.ndarray:
    """One octave of value noise over ``shape``, wrapping left to right."""
    height, width = shape
    rows, columns = cells
    lattice = _lattice(rows, columns, generator)
    y = np.linspace(0.0, rows, height, endpoint=False)
    x = np.linspace(0.0, columns, width, endpoint=False)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    fy = _smooth(y - y0)[:, None]
    fx = _smooth(x - x0)[None, :]
    y1 = np.minimum(y0 + 1, rows)
    x1 = x0 + 1
    top = lattice[y0][:, x0] * (1 - fx) + lattice[y0][:, x1] * fx
    bottom = lattice[y1][:, x0] * (1 - fx) + lattice[y1][:, x1] * fx
    return np.asarray(top * (1 - fy) + bottom * fy)


def _clouds(shape: tuple[int, int], generator: np.random.Generator,
            octaves: int = 5, cover: float = 0.48) -> np.ndarray:
    """Fair-weather cloud over the whole sphere, 0 clear to 1 solid.

    Summed octaves of value noise, then a soft threshold: below ``cover`` is
    open sky, and what is above it climbs to solid over a narrow band, which is
    what gives cloud an edge rather than a haze.
    """
    total = np.zeros(shape, dtype='f8')
    amplitude = 1.0
    weight = 0.0
    for octave in range(octaves):
        cells = (2 * 2 ** octave, 4 * 2 ** octave)
        total += amplitude * _noise(shape, cells, generator)
        weight += amplitude
        amplitude *= 0.5
    total /= weight
    return np.clip((total - cover) / 0.18, 0.0, 1.0)


def _elevation(height: int) -> np.ndarray:
    """The angle above the horizon of each row, +pi/2 at the top."""
    rows = (np.arange(height) + 0.5) / height
    return (0.5 - rows) * math.pi


def _azimuth(width: int) -> np.ndarray:
    """The compass angle of each column, one full turn across the image."""
    return ((np.arange(width) + 0.5) / width) * 2.0 * math.pi - math.pi


def panorama(width: int = 2048, height: int = 1024, seed: int = 11,
             sun_azimuth: float = 0.0, sun_elevation: float = math.radians(55.0),
             ground: tuple[float, float, float] = GROUND_COLOUR,
             horizon: tuple[float, float, float] = HORIZON_COLOUR,
             zenith: tuple[float, float, float] = ZENITH_COLOUR) -> Any:
    """An equirectangular sky: cloud above the horizon, ``ground`` below it.

    ``sun_azimuth`` and ``sun_elevation`` place the bright patch of sky, and are
    meant to be the direction the world's own sun comes *from* -- a panorama
    whose brightest quarter faces one way while the shadows fall the other is a
    picture of two afternoons.

    Returns a PIL image, which is what the glTF writer and anything looking at
    it both want.
    """
    from PIL import Image

    generator = np.random.default_rng(seed)
    elevation = _elevation(height)[:, None]
    azimuth = _azimuth(width)[None, :]

    above = elevation > 0.0
    # The gradient: horizon haze climbing to the zenith. Square-rooted so the
    # haze holds a band near the horizon rather than vanishing at once.
    climb = np.clip(np.sin(np.maximum(elevation, 0.0)), 0.0, 1.0) ** 0.6
    sky = (np.asarray(horizon)[None, None, :] * (1.0 - climb[:, :, None])
           + np.asarray(zenith)[None, None, :] * climb[:, :, None])

    # The sun's own glow: a broad, soft patch, brightest where it stands.
    angle = np.arccos(np.clip(
        np.sin(elevation) * math.sin(sun_elevation)
        + np.cos(elevation) * math.cos(sun_elevation) * np.cos(azimuth - sun_azimuth),
        -1.0, 1.0))
    glow = np.exp(-(angle / SUN_WIDTH) ** 2) * SUN_GLOW
    sky = sky + glow[:, :, None] * np.asarray((1.0, 0.95, 0.85))[None, None, :]

    # Cloud, thinning to nothing at the horizon so the band there stays haze.
    cover = _clouds((height, width), generator)
    cover = cover * np.clip(np.sin(np.maximum(elevation, 0.0)) * 3.0, 0.0, 1.0)
    lit = np.asarray((1.0, 0.99, 0.97))[None, None, :] * (0.72 + 0.28 * glow[:, :, None])
    sky = sky * (1.0 - cover[:, :, None]) + lit * cover[:, :, None]

    # Below the horizon: the room's floor, darkening downwards as a floor does.
    depth = np.clip(-np.sin(np.minimum(elevation, 0.0)), 0.0, 1.0)
    below = np.asarray(ground)[None, None, :] * (1.0 - 0.45 * depth[:, :, None])
    picture = np.where(above[:, :, None], sky, below)

    return Image.fromarray(
        np.clip(picture * 255.0 + 0.5, 0, 255).astype('u1'), mode='RGB')


def _jpeg(image: Any, quality: int = 88) -> bytes:
    """``image`` as JPEG bytes.

    A sky is a photograph rather than a diagram -- gradients and cloud, no flat
    areas and no text -- so JPEG costs a fifth of the PNG and shows nothing of
    it once the sky is a mile away or blurred into a reflection.
    """
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality,
                              optimize=True)
    return buffer.getvalue()


def _chunks(blob: bytes) -> tuple[dict, bytes]:
    """A glB's JSON document and its binary chunk's bytes (empty if none)."""
    magic, version, _length = struct.unpack('<III', blob[:12])
    if magic != _MAGIC or version != 2:
        raise ValueError('not a glTF 2.0 binary file')
    size, kind = struct.unpack('<II', blob[12:20])
    if kind != _JSON_CHUNK:
        raise ValueError('the first chunk of a glB is its JSON')
    document = json.loads(blob[20:20 + size])
    at = 20 + size
    if len(blob) >= at + 8:
        length, kind = struct.unpack('<II', blob[at:at + 8])
        if kind == _BIN_CHUNK:
            return document, blob[at + 8:at + 8 + length]
    return document, b''


def _rebuilt(document: dict, binary: bytes) -> bytes:
    """A glB carrying ``document`` and ``binary`` as its binary chunk."""
    body = json.dumps(document, separators=(',', ':')).encode('utf-8')
    body += b' ' * (-len(body) % 4)
    chunks = struct.pack('<II', len(body), _JSON_CHUNK) + body
    if binary:
        binary += b'\0' * (-len(binary) % 4)
        chunks += struct.pack('<II', len(binary), _BIN_CHUNK) + binary
    return struct.pack('<III', _MAGIC, 2, 12 + len(chunks)) + chunks


def _without_sky(document: dict, binary: bytes) -> bytes:
    """Take out a sky :func:`with_sky` wrote earlier; the binary chunk left.

    Its image, texture and sampler are the last of each, as this module
    appends them, and its bytes the last of the chunk; anything else in the
    document is left as it was.
    """
    skies = (document.get('extensions') or {}).pop(EXTENSION, None)
    if skies is None:
        return binary
    textures, images = document.get('textures', []), document.get('images', [])
    samplers, views = document.get('samplers', []), document.get('bufferViews', [])
    if textures and textures[-1].get('name') == 'Sky':
        texture = textures.pop()
        if texture.get('sampler') == len(samplers) - 1:
            samplers.pop()
        if texture.get('source') == len(images) - 1:
            image = images.pop()
            view = image.get('bufferView')
            if view is not None and view == len(views) - 1:
                binary = binary[:views.pop()['byteOffset']]
    return binary


def with_sky(blob: bytes, image: Any, ambient: float = 1.0) -> bytes:
    """``blob`` with ``image`` as the document's sky; the new glB.

    The picture goes in as an ordinary glTF image, in the glB's binary chunk
    behind a bufferView, with a texture and sampler, so a reader that has never
    heard of the extension sees a texture nothing uses and draws the world
    exactly as before. The extension is listed as *used* and never as
    required, for the same reason. A sky this wrote before is replaced.
    """
    document, binary = _chunks(blob)
    binary = _without_sky(document, binary)
    buffers = document.setdefault('buffers', [])
    if buffers and 'uri' in buffers[0]:
        raise ValueError('buffer 0 of this glB is a file of its own, not its '
                         'binary chunk')
    picture = _jpeg(image)
    binary += b'\0' * (-len(binary) % 4)
    views = document.setdefault('bufferViews', [])
    views.append({'buffer': 0, 'byteOffset': len(binary),
                  'byteLength': len(picture)})
    binary += picture
    if buffers:
        buffers[0]['byteLength'] = len(binary)
    else:
        buffers.append({'byteLength': len(binary)})

    images = document.setdefault('images', [])
    textures = document.setdefault('textures', [])
    samplers = document.setdefault('samplers', [])
    images.append({'bufferView': len(views) - 1, 'mimeType': 'image/jpeg',
                   'name': 'Sky'})
    # Linear filtering and a repeat in longitude: the panorama meets itself
    # there, and clamping it would draw the seam it was made to avoid.
    samplers.append({'magFilter': 9729, 'minFilter': 9987,
                     'wrapS': 10497, 'wrapT': 33071})
    textures.append({'source': len(images) - 1, 'sampler': len(samplers) - 1,
                     'name': 'Sky'})

    used = document.setdefault('extensionsUsed', [])
    if EXTENSION not in used:
        used.append(EXTENSION)
    document.setdefault('extensions', {})[EXTENSION] = {
        'skies': [{
            'type': 'panorama',
            'panorama': {'equirectangular': len(textures) - 1},
            'ambientSkyContribution': float(ambient),
        }],
    }
    return _rebuilt(document, binary)
