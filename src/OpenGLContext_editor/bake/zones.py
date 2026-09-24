"""A world's zones: a glTF document of ``OGLC_zone`` nodes beside the tileset.

A zone is a region with settings that hold inside it -- the environment
lighting captured in a tunnel, birdsong in a forest, surf beside a causeway,
the reverb of a bore. None of that belongs to one tile: a tunnel's zone runs
through as many tiles as the tunnel does. So a world's zones are written once,
as a small glTF document of zone nodes (with the shapes they name and the
sounds they play), and the tileset's ``extras.zones`` names it. OpenGLContext's
:class:`~OpenGLContext.scenegraph.tilesterrain.TilesTerrain` loads it and keeps
it mounted, and the render pass does the rest (``docs/zones.rst``).

:class:`ZoneRecord` is one zone, in the world's own coordinates.
:class:`AmbientSound` is a looping sound zones may play, carried as a WAV file
beside the document. :class:`ZonesLayer` is the bake layer that writes them:
no content in any tile, a record in the tileset's extras, and the document and
its sounds as assets. :func:`zone_records` turns a road's
:class:`~OpenGLContext_editor.world.places.Place` list into records, with each
kind of place's environment, reverb and ambience.
"""
from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.places import (
    BRIDGE, CAUSEWAY, FOREST, TUNNEL, Place,
)

__all__ = [
    'DOCUMENT', 'AmbientSound', 'ZoneRecord', 'ZonesLayer', 'PLACE_SETTINGS',
    'zone_records', 'place_sounds', 'wav_bytes',
]

#: What the zones document is called beside the tileset.
DOCUMENT = 'zones.gltf'

#: The rate ambience is written at: enough for birdsong's top, and half the
#: bytes of CD rate.
SAMPLE_RATE = 32000


@dataclass(frozen=True)
class AmbientSound:
    """A looping sound zones play, heard while the camera is in them.

    ``make`` returns the samples, mono float in -1..1 at :data:`SAMPLE_RATE`;
    ``gain`` is the emitter's level. Written into the world as
    ``audio/<name>.wav``.
    """

    name: str
    make: Callable[[], Any]
    gain: float = 1.0

    @property
    def uri(self) -> str:
        return 'audio/%s.wav' % (self.name,)


@dataclass(frozen=True)
class ZoneRecord:
    """One zone: a box placed in the world, and what holds inside it.

    ``centre`` and ``rotation`` (a glTF quaternion) place the box, whose full
    extents are ``size``. ``environment`` and ``reverb`` are the zone's own
    blocks as ``OGLC_zone`` writes them, and ``sounds`` the names of the
    :class:`AmbientSound` emitters it plays.
    """

    name: str
    centre: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    size: tuple[float, float, float]
    priority: int = 0
    blend: float = 0.0
    environment: Optional[dict] = None
    reverb: Optional[dict] = None
    sounds: tuple[str, ...] = ()


def wav_bytes(samples: Any, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono float samples as a 16-bit PCM WAV file."""
    pcm = (np.clip(np.asarray(samples, dtype='d'), -1.0, 1.0) * 32767.0).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def place_sounds(seed: int = 0) -> dict[str, AmbientSound]:
    """The ambience a road's places play, made rather than recorded.

    Birdsong in the forest and surf beside a causeway, each a loop that
    repeats without a seam (:func:`omi_audio.synth.birdsong`,
    :func:`omi_audio.synth.surf`). A bore has no ambience of its own: what is
    heard in it is the car, given back by the bore's reverb.
    """
    from omi_audio import synth

    def birds() -> Any:
        return synth.birdsong(16.0, sample_rate=SAMPLE_RATE, seed=seed + 1).samples

    def waves() -> Any:
        return synth.surf(14.0, sample_rate=SAMPLE_RATE, seed=seed + 2, waves=4).samples

    return {'birdsong': AmbientSound('birdsong', birds, gain=0.45),
            'surf': AmbientSound('surf', waves, gain=0.55)}


#: What each kind of place holds. Every place captures its own environment
#: from the road, so a bore is lit and reflected by its own walls and lamps,
#: a forest by its trees and a causeway by its water and open sky. The bore
#: gives back a long, dark reverb.
PLACE_SETTINGS: dict[str, dict] = {
    TUNNEL: {'reverb': {'level': 0.45, 'decay': 1.9, 'damping': 0.5},
             'sounds': ()},
    CAUSEWAY: {'sounds': ('surf',)},
    BRIDGE: {'sounds': ()},
    FOREST: {'sounds': ('birdsong',)},
}


def zone_records(places: Sequence[Place],
                 settings: Optional[dict[str, dict]] = None) -> list[ZoneRecord]:
    """A zone for each place, with its kind's settings (:data:`PLACE_SETTINGS`)."""
    chosen = PLACE_SETTINGS if settings is None else settings
    counts: dict[str, int] = {}
    records = []
    for place in places:
        kind = chosen.get(place.kind, {})
        counts[place.kind] = counts.get(place.kind, 0) + 1
        rules = place.rules
        records.append(ZoneRecord(
            name='%s-%d' % (place.kind, counts[place.kind]),
            centre=place.centre, rotation=place.rotation, size=place.size,
            priority=rules.priority, blend=rules.blend,
            environment={'capture': {'position': list(place.eye)}},
            reverb=kind.get('reverb'), sounds=tuple(kind.get('sounds', ()))))
    return records


@dataclass
class ZonesLayer:
    """The zones of a world, written beside its tileset (see the module docstring)."""

    zones: Sequence[ZoneRecord]
    sounds: dict[str, AmbientSound] = field(default_factory=dict)
    name: str = 'zones'

    def bounds(self) -> BoundingBox | None:
        """Nothing: zones are not tile content, and hold no region of the bake open."""
        return None

    def content(self, region: BoundingBox, error: float) -> list:
        """Nothing in any tile."""
        return []

    def metadata(self) -> dict[str, Any]:
        """The record the tileset carries: where the zones document is."""
        if not self.zones:
            return {}
        return {'zones': {'document': DOCUMENT, 'count': len(self.zones)}}

    def assets(self) -> dict[str, bytes]:
        """The zones document, and every sound a zone plays."""
        if not self.zones:
            return {}
        written = {DOCUMENT: (json.dumps(self.document(), indent=1) + '\n').encode('utf-8')}
        for sound in self._played():
            written[sound.uri] = wav_bytes(sound.make())
        return written

    def _played(self) -> list[AmbientSound]:
        wanted = []
        for record in self.zones:
            for name in record.sounds:
                sound = self.sounds.get(name)
                if sound is None:
                    raise ValueError('zone %r plays %r, which is not one of the '
                                     'sounds given' % (record.name, name))
                if sound not in wanted:
                    wanted.append(sound)
        return wanted

    def document(self) -> dict[str, Any]:
        """The zones as a glTF 2.0 document, with ``KHR_implicit_shapes`` shapes."""
        played = self._played()
        emitter = {sound.name: index for index, sound in enumerate(played)}
        nodes = []
        shapes = []
        for record in self.zones:
            block: dict[str, Any] = {'shape': len(shapes), 'priority': record.priority,
                                     'blend': record.blend}
            shapes.append({'type': 'box', 'box': {'size': list(record.size)}})
            if record.environment is not None:
                block['environment'] = record.environment
            if record.reverb is not None:
                block['reverb'] = record.reverb
            if record.sounds:
                block['extensions'] = {'KHR_audio_emitter': {
                    'emitters': [emitter[name] for name in record.sounds]}}
            nodes.append({'name': record.name, 'translation': list(record.centre),
                          'rotation': list(record.rotation),
                          'extensions': {'OGLC_zone': block}})
        used = ['OGLC_zone', 'KHR_implicit_shapes']
        extensions: dict[str, Any] = {'KHR_implicit_shapes': {'shapes': shapes}}
        if played:
            used.append('KHR_audio_emitter')
            extensions['KHR_audio_emitter'] = {
                'audio': [{'uri': sound.uri, 'mimeType': 'audio/wav'} for sound in played],
                'sources': [{'name': sound.name, 'audio': index, 'loop': True,
                             'autoplay': True}
                            for index, sound in enumerate(played)],
                'emitters': [{'name': sound.name, 'type': 'global', 'gain': sound.gain,
                              'sources': [index]}
                             for index, sound in enumerate(played)],
            }
        return {
            'asset': {'version': '2.0', 'generator': 'OpenGLContext_editor zones'},
            'extensionsUsed': used,
            'extensions': extensions,
            'scene': 0,
            'scenes': [{'nodes': list(range(len(nodes)))}],
            'nodes': nodes,
        }
