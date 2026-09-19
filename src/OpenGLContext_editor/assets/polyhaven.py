"""Fetching a published plant, and the maps a cutout one needs beside it.

`Poly Haven <https://polyhaven.com/>`_ publishes scanned models under **CC0**,
which asks for no attribution and places no condition on redistribution -- so a
world baked from one can ship the result. Credit is given anyway, in
:func:`credit`, because it is the decent thing and costs a line.

    from OpenGLContext_editor.assets import polyhaven

    source = polyhaven.fetch('fern_02', 'downloads')
    species = plants.bake(source, 'assets')

**The glTF download is not enough on its own.** Its base colour is published as
a JPEG, and a JPEG has no alpha channel, so a plant made of alpha-cut cards
arrives with its cutout missing -- the model would draw as a fan of opaque
rectangles. The mask is published separately, under whichever of ``Alpha``,
``opacity`` or ``Mask`` that asset happens to use, and :func:`fetch` brings it
along so the bake can put it back.

**Nothing is fetched twice.** Downloads and the library's own answers are kept
in a per-user cache -- :func:`cache_dir`, beside the rest of OpenGLContext's
cached assets -- so re-baking, or baking the same plant into a second world,
asks Poly Haven for nothing. They give this work away; hammering them for bytes
already on the disk is not a way to say thank you. ``OPENGLCONTEXT_POLYHAVEN``
names somewhere else, and a caller may pass a directory of its own.

Nothing here is needed at play time.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from OpenGLContext_editor.assets.plants import PlantSource

__all__ = ['API', 'MASK_KEYS', 'cache_dir', 'fetch', 'credit', 'info', 'files']

#: Where downloads are kept when a caller names nowhere. Read once per call
#: rather than at import, so a test or a script can point it elsewhere.
CACHE_VARIABLE = 'OPENGLCONTEXT_POLYHAVEN'

#: Where the library answers questions about itself.
API = 'https://api.polyhaven.com'

#: What an asset may call its cutout mask, best first. The name is the
#: publisher's choice per asset rather than a convention, so all three have to
#: be looked for; solid geometry has none of them and needs none.
MASK_KEYS = ('Alpha', 'opacity', 'Mask')

#: Which encoding of a mask to take. PNG first: a mask is a hard edge and JPEG
#: ringing along one shows as a fringe of half-transparent pixels round every
#: leaf.
MASK_FORMATS = ('png', 'jpg')

#: A default that names the caller, because the library answers an anonymous
#: request with a refusal.
USER_AGENT = ('OpenGLContext-editor '
              '(+https://github.com/mcfletch/openglcontext-editor)')

#: Where Poly Haven publishes from: the library that answers questions, and the
#: host its files come from. The answers name the URLs to fetch, and an answer
#: is not permission to fetch from anywhere -- see
#: :func:`OpenGLContext.loaders.resolver.require_host`.
HOSTS = ('api.polyhaven.com', 'dl.polyhaven.org', 'polyhaven.com')

#: Ceiling on one fetched file. An 8k model with its maps is tens of megabytes;
#: this is room for the largest thing published and a bound on the rest.
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024

Transport = Callable[[str], bytes]


def cache_dir(directory: str | None = None) -> str:
    """Where fetched assets are kept, per user rather than in shared temp.

    The same place the rest of OpenGLContext keeps downloaded assets, so no
    other account can pre-seed a file this user then bakes into a world.
    """
    if directory is None:
        directory = os.environ.get(CACHE_VARIABLE) or ''
    if not directory:
        from OpenGLContext import userpaths
        directory = os.path.join(userpaths.appdatadirectory(), 'OpenGLContext',
                                 'polyhaven')
    os.makedirs(directory, mode=0o700, exist_ok=True)
    return directory


def _open_capped(url: str, max_bytes: int) -> bytes:
    """``url``'s bytes, reading no more than ``max_bytes`` of them."""
    from urllib.request import Request, urlopen

    from OpenGLContext.loaders import resolver
    with urlopen(Request(url, headers={'User-Agent': USER_AGENT}),  # noqa: S310
                 timeout=120) as answer:
        return resolver.stream_capped(answer, max_bytes)


def _get(url: str) -> bytes:
    """One URL's bytes, over the network, from somewhere Poly Haven publishes.

    The URLs fetched here come out of the library's own answers, so they are
    checked before a socket is opened rather than trusted for having arrived
    over TLS: an answer says where a file is, and that is a claim.
    """
    from OpenGLContext.loaders import resolver
    return _open_capped(resolver.require_host(url, HOSTS), MAX_DOWNLOAD_BYTES)


def _asked(kind: str, slug: str, transport: Transport | None,
           directory: str | None) -> dict:
    """One of the library's answers about ``slug``, from the cache or the wire.

    The answers change when an asset is re-published, which is rare, and a bake
    of eight plants would otherwise ask sixteen questions every time it ran.
    """
    where = os.path.join(cache_dir(directory), '_api',
                         '%s-%s.json' % (kind, slug))
    if os.path.exists(where) and os.path.getsize(where):
        with open(where, encoding='utf-8') as handle:
            return dict(json.load(handle))
    answer = json.loads((transport or _get)('%s/%s/%s' % (API, kind, slug)))
    os.makedirs(os.path.dirname(where), exist_ok=True)
    with open(where, 'w', encoding='utf-8') as handle:
        json.dump(answer, handle)
    return dict(answer)


def info(slug: str, transport: Transport | None = None,
         directory: str | None = None) -> dict:
    """What the library says an asset is: its name, its authors, its size."""
    return _asked('info', slug, transport, directory)


def files(slug: str, transport: Transport | None = None,
          directory: str | None = None) -> dict:
    """Every file published for an asset, by kind and resolution."""
    return _asked('files', slug, transport, directory)


def credit(slug: str, described: dict | None = None,
           transport: Transport | None = None,
           directory: str | None = None) -> str:
    """One line naming an asset and who made it.

    CC0 requires none of this. It is written because the work was given away
    and saying whose it was costs nothing.
    """
    described = (described if described is not None
                 else info(slug, transport, directory))
    authors = list((described.get('authors') or {}).keys())
    if not authors:
        who = 'Poly Haven'
    elif len(authors) == 1:
        who = authors[0]
    else:
        who = '%s and %s' % (', '.join(authors[:-1]), authors[-1])
    return "'%s' by %s, from Poly Haven (%s), CC0" % (
        described.get('name', slug), who, 'https://polyhaven.com/a/%s' % (slug,))


@dataclass(frozen=True)
class Download:
    """What was fetched for one asset, and where it was put."""

    slug: str
    directory: str
    gltf: str
    diffuse: str
    mask: str | None
    credit: str

    def source(self) -> PlantSource:
        """This download, as the bake wants to be told about it."""
        return PlantSource(slug=self.slug, gltf=self.gltf,
                           diffuse=self.diffuse, mask=self.mask,
                           credit=self.credit)


def _under(directory: str, relative: str) -> str:
    """Where ``relative`` lands under ``directory``, refusing to leave it.

    The layout of a download is the library's to describe -- ``textures/x.jpg``
    beside the model that names it -- so the names come from the answer, and a
    name is data. ``../../.bashrc`` as a key would otherwise write bytes the
    same answer chose wherever this process can write.
    """
    from OpenGLContext.loaders.resolver import Resolver
    return Resolver(base_dir=directory).resolve(relative)


def _save(url: str, path: str, transport: Transport | None) -> str:
    """``url`` at ``path``, unless it is already there."""
    if os.path.exists(path) and os.path.getsize(path):
        return path
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    data = (transport or _get)(url)
    with open(path, 'wb') as handle:
        handle.write(data)
    return path


def _mask_url(published: dict, resolution: str) -> str | None:
    """Where an asset's cutout mask is, or nothing where it has none."""
    for key in MASK_KEYS:
        sizes = published.get(key)
        if not sizes:
            continue
        encodings = sizes.get(resolution) or {}
        for kind in MASK_FORMATS:
            if kind in encodings:
                return str(encodings[kind]['url'])
    return None


def fetch(slug: str, directory: str | None = None, resolution: str = '1k',
          transport: Transport | None = None) -> Download:
    """Download ``slug`` into ``directory`` and say what came.

    Takes the glTF at ``resolution`` with the files it names, and the cutout
    mask from beside it. ``directory`` defaults to the shared per-user
    :func:`cache_dir`; whatever is already there is left alone, so a re-bake
    and a second world asking for the same plant both cost nothing.
    """
    directory = cache_dir(directory)
    published = files(slug, transport, directory)
    if 'gltf' not in published:
        raise LookupError(
            "%s publishes no glTF; this bakes models, not materials" % (slug,))
        # A material -- a ground texture, say -- has maps and no geometry.
    sizes = published['gltf']
    if resolution not in sizes:
        raise LookupError("%s is not published at %s, only at %s"
                          % (slug, resolution, ', '.join(sorted(sizes))))
    entry = sizes[resolution]['gltf']
    here = os.path.join(directory, slug)
    document = _save(entry['url'], _under(here, os.path.basename(
        entry['url'])), transport)
    diffuse = None
    for relative, named in (entry.get('include') or {}).items():
        saved = _save(named['url'], _under(here, relative), transport)
        if '_diff' in relative or 'diffuse' in relative.lower():
            diffuse = saved
    if diffuse is None:
        raise LookupError("%s names no base colour among %s"
                          % (slug, ', '.join(entry.get('include') or ())))
    url = _mask_url(published, resolution)
    mask = (None if url is None
            else _save(url, _under(here, os.path.basename(url)), transport))
    return Download(slug=slug, directory=here, gltf=document, diffuse=diffuse,
                    mask=mask,
                    credit=credit(slug, transport=transport, directory=directory))


def described(slug: str, transport: Transport | None = None,
              directory: str | None = None) -> dict:
    """A short summary of an asset, for a caller choosing between them."""
    found: dict[str, Any] = info(slug, transport, directory)
    return {'slug': slug, 'name': found.get('name', slug),
            'triangles': found.get('polycount'),
            'authors': list((found.get('authors') or {}).keys()),
            'categories': found.get('categories') or [],
            'credit': credit(slug, found)}
