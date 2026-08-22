"""A circuit's start/finish gantry, written into the tiles and the tileset.

The frame and the line painted under it go into the tile that holds them as one
mesh reading one picture, so the marker a lap is measured against costs a world
a single draw.

The legs also travel in the tileset's ``extras``, as props. Tile geometry is
level-of-detail geometry that arrives and leaves as a camera moves, and a car
that hits a gantry leg has to hit it whatever the streamer is doing, so the
bodies are stood up from the tileset by
:class:`OpenGLContext.physics.props.PropColliders` -- the same channel the
boulders travel in.

*Where* the gantry belongs is decided in
:mod:`OpenGLContext_editor.world.gantry`, from the road's own alignment. The
object itself is :mod:`OpenGLContext.scenegraph.gantry`.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import ExternalImage, SceneNode
from OpenGLContext.scenegraph.gantry import (
    GantryProfile,
    gantry_atlas,
    gantry_legs,
    gantry_material,
    gantry_mesh,
    start_line_mesh,
)
from OpenGLContext.scenegraph.props import Prop

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.placing import gathered, placed
from OpenGLContext_editor.world.gantry import StartFinish

#: Where a world keeps its gantry artwork, relative to the tileset.
GANTRY_DIRECTORY = 'gantry'

#: The one image the steel, the banner and the painted line read out of.
ATLAS_IMAGE = '%s/marks.png' % (GANTRY_DIRECTORY,)

#: How big one part of that image is, in pixels.
CELL_PIXELS = 256

#: The coarsest tile that carries the gantry. It stands nine metres wide and
#: seven tall over the road, and a driver wants to see the line coming, so it
#: survives to a coarser tile than a sign does.
MAXIMUM_ERROR = 32.0

#: What the legs are called in the world's props.
LEG_KIND = 'gantry-leg'


@dataclass
class GantryLayer:
    """The start/finish marker of a circuit.

    ``placement`` is what :func:`~OpenGLContext_editor.world.gantry.start_finish`
    worked out from the road.
    """

    placement: StartFinish
    profile: GantryProfile = field(default_factory=GantryProfile)
    cell_pixels: int = CELL_PIXELS
    maximum_error: float = MAXIMUM_ERROR
    name: str = 'gantry'
    _mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        material = gantry_material(image=ExternalImage(ATLAS_IMAGE, srgb=True),
                                   cell=self.cell_pixels)
        cells = gantry_atlas(self.cell_pixels)[1]
        one = self.placement
        self._mesh = gathered(
            [placed(gantry_mesh(one.span, self.profile, material, cells,
                                one.drops), one.position, one.yaw),
             placed(start_line_mesh(one.width, self.profile, material, cells,
                                    one.crossfall), one.position, one.yaw,
                    roll=math.atan(one.bank))],
            material)

    def bounds(self) -> BoundingBox:
        """Everything the gantry occupies, whichever way it is turned."""
        at = np.asarray(self.placement.position, dtype='d')
        reach = self.placement.span / 2.0 + self.profile.leg_radius
        return BoundingBox(
            (at[0] - reach, at[1] - max(self.placement.drops), at[2] - reach),
            (at[0] + reach, at[1] + self.profile.height, at[2] + reach))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        if error > self.maximum_error or not _inside(self.placement.position,
                                                     region):
            return []
        return [SceneNode(mesh=self._mesh, name=self.name)]

    def assets(self) -> dict[str, bytes]:
        """The one picture the whole marker reads out of."""
        buffer = io.BytesIO()
        gantry_atlas(self.cell_pixels)[0].save(buffer, format='PNG')
        return {ATLAS_IMAGE: buffer.getvalue()}

    def metadata(self) -> dict[str, Any]:
        """Where the lap begins, and the two legs a car can hit.

        A game times a lap against the line, and a pile of triangles does not
        say where the line is. The legs go in the world's props because that is
        the channel a body is stood up from.
        """
        one = self.placement
        return {
            'start': {'at': [round(float(v), 3) for v in one.position],
                      'yaw': round(float(one.yaw), 4),
                      'span': round(float(one.span), 3),
                      'width': round(float(one.width), 3)},
            'props': [leg.to_json() for leg in self.legs()]}

    def legs(self) -> list[Prop]:
        """The two uprights as props, so a physics world can stand them up."""
        one = self.placement
        beam = np.array([math.cos(one.yaw), 0.0, -math.sin(one.yaw)])
        at = np.asarray(one.position, dtype='d')
        found = []
        for leg, drop in zip(gantry_legs(one.span, self.profile, one.drops),
                             one.drops, strict=True):
            foot = at + beam * leg.offset - np.array([0.0, drop, 0.0])
            found.append(Prop(kind=LEG_KIND, position=tuple(float(v)
                                                            for v in foot),
                              yaw=float(one.yaw), radius=leg.radius,
                              height=leg.height))
        return found


def _inside(position: Any, region: BoundingBox) -> bool:
    at = np.asarray(position, dtype='d')
    return bool(np.all(at >= region.minimum) and np.all(at <= region.maximum))
