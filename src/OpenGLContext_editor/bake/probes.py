"""Bake each zone's environment into the world, so a run draws none.

A baked world's zones ask for an environment captured inside them
(:mod:`OpenGLContext_editor.bake.zones`). Captured at run time, each one is six
draws of the whole world, repeated for every bounce, while the game is being
played. This captures them once, when the world is baked, and writes what the
capture made into the world as ``EXT_lights_image_based`` lights: a zone then
names its light, and the renderer uploads it with nothing drawn.

The capture is the engine's
(:func:`~OpenGLContext.passes.zonebake.bake_zone_lights`): the baked world is
opened in an offscreen context that asks for the PBR pass and the full probe
on itself, the tiles around each zone's capture point are streamed in before
every frame, and what each zone captured comes back as irradiance faces and
prefiltered mips. This module writes them: the mips become RGBD PNG faces,
the irradiance cube nine spherical-harmonic coefficients
(:func:`~OpenGLContext.scenegraph.imagebasedlight.sh_fit`), and the zones
document names each zone's light, found by the zone node's name.

It needs a GPU the offscreen context can open (EGL on Linux). Where there is
none, :func:`bake_probes` says so and leaves the zones capturing at run time.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
from OpenGLContext import atomicfiles

log = logging.getLogger(__name__)

__all__ = ['bake_probes', 'PROBE_DIRECTORY', 'EXTENSION']

EXTENSION = 'EXT_lights_image_based'
#: Where the probe faces are written, beside the zones document.
PROBE_DIRECTORY = 'probes'


def _png(values: np.ndarray) -> bytes:
    from OpenGLContext.scenegraph.imagebasedlight import encode_rgbd
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(encode_rgbd(values)).save(buffer, 'PNG')
    return buffer.getvalue()


def bake_probes(directory: str, document: str = 'zones.gltf',
                size: tuple[int, int] = (256, 256),
                scene: Callable[[Any], list] | None = None,
                progress: Callable[[int, int], None] | None = None) -> int:
    """How many zones of the world in ``directory`` were baked and written.

    ``scene`` gives the nodes to light the world with around the terrain --
    a sky, a sun -- and defaults to the engine viewer's own. Nought where no
    offscreen context could be opened or the world has no zones, which leaves
    them capturing at run time.
    """
    path = os.path.join(directory, document)
    if not os.path.exists(path) or not os.path.exists(
            os.path.join(directory, 'tileset.json')):
        return 0
    try:
        from OpenGLContext.eglcontext import EGLContext
    except (ImportError, OSError) as error:   # pragma: no cover - no EGL here
        log.warning('no offscreen context (%s); zones capture at run time', error)
        return 0
    from OpenGLContext.passes.zonebake import bake_zone_lights
    from OpenGLContext.scenegraph.basenodes import sceneGraph
    from OpenGLContext.scenegraph.tilesterrain import TilesTerrain

    terrain = TilesTerrain(os.path.join(directory, 'tileset.json'), workers=2)
    if terrain.zones is None or not terrain.zones.zones:
        terrain.shutdown()
        return 0

    def around(world: Any) -> list:
        from OpenGLContext.viewer.environment import horizon_background
        from OpenGLContext.viewer.sceneviewer import ViewerContext
        return [horizon_background(), *ViewerContext.defaultLights(1000.0)]

    lighting = (scene or around)(terrain)

    class Baker(EGLContext):
        renderer = 'pbr'
        profile = 'core'

        def OnInit(self) -> None:
            self.sg = sceneGraph(children=[*lighting, terrain])

    def stream(eye: tuple[float, float, float]) -> None:
        terrain.update_for_camera(np.asarray(eye, 'd'), size[1])
        terrain.wait_for_loads(timeout=5.0)

    names = _zone_names(terrain.zones)
    try:
        with Baker(size=size, ibl='full') as context:
            baked = bake_zone_lights(context, before_frame=stream, progress=progress)
    finally:
        terrain.shutdown()
    written: dict[str, bytes] = {}
    lights: dict[str, dict] = {}
    for one in baked:
        name = names.get(id(one.zone))
        if not name:
            log.warning('a baked zone has no node name to be found by; it '
                        'captures at run time')
            continue
        lights[name] = _light(name, one.irradiance, one.mips, written)
    if not lights:
        return 0
    for relative, data in written.items():
        atomicfiles.write_bytes(os.path.join(directory, relative), data)
    _rewrite(path, lights)
    return len(lights)


def _zone_names(scene: Any) -> dict[int, str]:
    """Each ``Zone`` node of a loaded zones document, by id, to the name of the
    glTF node that carries it."""
    from OpenGLContext.scenegraph.zone import Zone
    names = {}
    for index, transform in scene.node_transforms.items():
        name = scene.node_names.get(index)
        for child in getattr(transform, 'children', ()):
            if isinstance(child, Zone) and name:
                names[id(child)] = name
    return names


def _light(name: str, irradiance: Any, mips: Any, written: dict[str, bytes]) -> dict:
    """One zone's ``EXT_lights_image_based`` light, its faces added to ``written``."""
    from OpenGLContext.scenegraph.imagebasedlight import sh_fit
    images = []
    for level, faces in enumerate(mips):
        level_names = []
        for face, pixels in enumerate(faces):
            relative = '%s/%s-m%d-f%d.png' % (PROBE_DIRECTORY, name, level, face)
            written[relative] = _png(pixels)
            level_names.append(relative)
        images.append(level_names)
    coefficients = sh_fit([face * math.pi for face in irradiance])
    return {'name': name, 'irradianceCoefficients': coefficients.tolist(),
            'specularImageSize': int(mips[0][0].shape[0]),
            'specularImages': images}


def _rewrite(path: str, lights: dict[str, dict]) -> None:
    """Point each baked zone, found by its node's name, at its light.

    A zone that already names a light from an earlier bake has that light and
    its images replaced; every other zone's light is kept.
    """
    with open(path, encoding='utf-8') as handle:
        doc = json.load(handle)
    images = doc.setdefault('images', [])
    existing = doc.setdefault('extensions', {}).setdefault(EXTENSION, {}).setdefault(
        'lights', [])
    dropped: set[int] = set()
    for node in doc.get('nodes', []):
        block = (node.get('extensions') or {}).get('OGLC_zone')
        light = lights.get(node.get('name'))
        if block is None or light is None:
            continue
        light = dict(light, specularImages=[
            [_image(images, uri) for uri in level] for level in light['specularImages']])
        named = (block.get('extensions') or {}).get(EXTENSION) or {}
        index = named.get('light')
        if isinstance(index, int) and 0 <= index < len(existing):
            dropped.update(i for level in existing[index].get('specularImages', ())
                           for i in level)
            existing[index] = light
        else:
            index = len(existing)
            existing.append(light)
        environment = dict(block.get('environment') or {})
        environment.pop('capture', None)
        block['environment'] = environment
        block.setdefault('extensions', {})[EXTENSION] = {'light': index}
    if dropped:
        kept = [i for i in range(len(images)) if i not in dropped]
        moved = {old: new for new, old in enumerate(kept)}
        doc['images'] = [images[i] for i in kept]
        for light in existing:
            light['specularImages'] = [[moved[i] for i in level]
                                       for level in light['specularImages']]
    doc['extensionsUsed'] = sorted(set(doc.get('extensionsUsed', [])) | {EXTENSION})
    atomicfiles.write_text(path, json.dumps(doc, indent=1) + '\n')


def _image(images: list, uri: str) -> int:
    images.append({'uri': uri, 'mimeType': 'image/png'})
    return len(images) - 1
