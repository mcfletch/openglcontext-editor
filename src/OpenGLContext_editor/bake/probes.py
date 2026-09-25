"""Bake each zone's environment into the world, so a run draws none.

A baked world's zones ask for an environment captured inside them
(:mod:`OpenGLContext_editor.bake.zones`). Captured at run time, each one is six
draws of the whole world, repeated for every bounce, while the game is being
played. This captures them once, when the world is baked, and writes what the
capture made into the world as ``EXT_lights_image_based`` lights: a zone then
names its light, and the renderer uploads it with nothing drawn.

The capture is the engine's own
(:meth:`~OpenGLContext.passes.zonepass.ZonesMixin.renderZoneProbes`): the
baked world is opened in an offscreen context, the camera is stood at each
zone's capture point with the tiles around it streamed in, and frames are
drawn until the zone has had every capture it asks for. Its probe layer is
read back (:meth:`~OpenGLContext.passes.ibl.IBLProbe.read_layer`): the
prefiltered mips become RGBD PNG faces, and the irradiance cube becomes nine
spherical-harmonic coefficients (:func:`~OpenGLContext.scenegraph.imagebasedlight.sh_fit`).

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

log = logging.getLogger(__name__)

__all__ = ['bake_probes', 'PROBE_DIRECTORY', 'EXTENSION']

EXTENSION = 'EXT_lights_image_based'
#: Where the probe faces are written, beside the zones document.
PROBE_DIRECTORY = 'probes'
#: How many frames one zone is given to stream its surroundings and finish
#: its captures before it is written with what it has.
FRAMES_PER_ZONE = 40


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
    """Capture every zone of the world in ``directory`` and write its light.

    ``scene`` gives the nodes to light the world with around the terrain --
    a sky, a sun -- and defaults to the engine viewer's own. Returns how many
    zones were baked; nought where no offscreen context could be opened or
    the world has no zones, which leaves them capturing at run time.
    """
    path = os.path.join(directory, document)
    if not os.path.exists(path):
        return 0
    # The capture is the PBR pass's, with the full probe it fills layers of.
    os.environ['OPENGLCONTEXT_RENDERER'] = 'pbr'
    os.environ['OPENGLCONTEXT_PROFILE'] = 'core'
    os.environ['OPENGLCONTEXT_IBL'] = 'full'
    try:
        from OpenGLContext.eglcontext import EGLContext
    except (ImportError, OSError) as error:   # pragma: no cover - no EGL here
        log.warning('no offscreen context (%s); zones capture at run time', error)
        return 0
    from OpenGLContext.passes import renderpass
    from OpenGLContext.scenegraph.basenodes import sceneGraph
    from OpenGLContext.scenegraph.imagebasedlight import sh_fit
    from OpenGLContext.scenegraph.tilesterrain import TilesTerrain
    from OpenGLContext.scenegraph.zone import ENVIRONMENT

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
        def OnInit(self) -> None:
            self.sg = sceneGraph(children=[*lighting, terrain])

    zones = list(terrain.zones.zones)
    written: dict[str, bytes] = {}
    lights: list[dict] = []
    by_zone: dict[int, int] = {}
    try:
        with Baker(size=size) as context:
            context.OnDraw(force=1)
            flat = renderpass.current_pass()
            if flat is None:
                raise RuntimeError('the bake context drew no render pass')
            for index, zone in enumerate(zones):
                placed = {id(p.zone): p for p in flat.zones}.get(id(zone))
                setting = zone.setting(ENVIRONMENT)
                if placed is None or setting is None or not bool(setting.capture):
                    continue
                eye = placed.to_world(setting.captureCentre)
                context.platform.setPosition(tuple(float(v) for v in eye))
                key = id(zone)
                for _frame in range(FRAMES_PER_ZONE):
                    terrain.update_for_camera(np.asarray(eye, 'd'), size[1])
                    terrain.wait_for_loads(timeout=5.0)
                    context.OnDraw(force=1)
                    schedule = flat._zoneCaptures
                    if (schedule is not None and schedule.captured(key) >= schedule.bounces + 1
                            and not schedule.waiting):
                        break
                schedule = flat._zoneCaptures
                layer = None if schedule is None else schedule.layer(key)
                if layer is None:
                    log.warning('zone %d was not captured; it captures at run time', index)
                    continue
                irradiance, mips = flat._ibl_probe.read_layer(layer)
                light = len(lights)
                images = []
                for level, faces in enumerate(mips):
                    names = []
                    for face, pixels in enumerate(faces):
                        name = '%s/zone%d-m%d-f%d.png' % (PROBE_DIRECTORY, light, level, face)
                        written[name] = _png(pixels)
                        names.append(name)
                    images.append(names)
                coefficients = sh_fit([face * math.pi for face in irradiance])
                lights.append({'name': 'zone-%d' % (index,),
                               'irradianceCoefficients': coefficients.tolist(),
                               'specularImageSize': int(mips[0][0].shape[0]),
                               'specularImages': images})
                by_zone[index] = light
                if progress is not None:
                    progress(index + 1, len(zones))
    finally:
        terrain.shutdown()
    if not lights:
        return 0
    for name, data in written.items():
        target = os.path.join(directory, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'wb') as handle:
            handle.write(data)
    _rewrite(path, by_zone, lights)
    return len(lights)


def _rewrite(path: str, by_zone: dict[int, int], lights: list[dict]) -> None:
    """Point each baked zone at its light, and add the lights and their images."""
    with open(path) as handle:
        doc = json.load(handle)
    images = doc.setdefault('images', [])
    for light in lights:
        light['specularImages'] = [
            [_image(images, name) for name in level] for level in light['specularImages']]
    doc.setdefault('extensions', {})[EXTENSION] = {'lights': lights}
    used = set(doc.get('extensionsUsed', [])) | {EXTENSION}
    doc['extensionsUsed'] = sorted(used)
    zone_nodes = [node for node in doc.get('nodes', [])
                  if 'OGLC_zone' in (node.get('extensions') or {})]
    for index, node in enumerate(zone_nodes):
        if index not in by_zone:
            continue
        block = node['extensions']['OGLC_zone']
        environment = dict(block.get('environment') or {})
        environment.pop('capture', None)
        block['environment'] = environment
        block.setdefault('extensions', {})[EXTENSION] = {'light': by_zone[index]}
    with open(path, 'w') as handle:
        handle.write(json.dumps(doc, indent=1) + '\n')


def _image(images: list, uri: str) -> int:
    images.append({'uri': uri, 'mimeType': 'image/png'})
    return len(images) - 1
