"""Rivers in a baked world.

A channel carved into the terrain is a **valley**; what makes it a river is
water in it, and the carving is already the terrain's business -- the bed is an
edit on the height source, so every tile that meshes the ground meshes the bed.
What is missing until this layer is the surface, and a world nobody can drive to
the water of is a world with no rivers in it.

**The detail matters as much as the water.** A river is a few metres across. A
tile a kilometre away covers hundreds of metres and draws it two pixels wide,
and a full swept surface there spends the tile's whole budget on something
nobody can resolve -- while what the eye actually gets at that range is the sun
catching the surface here and there between whatever stands over it. So a near
tile gets the ribbon and a far one gets **glints**, which is
:func:`~OpenGLContext.scenegraph.water.water_glints`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import SceneNode
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.water import (
    FLOWING,
    WaterStyle,
    water_glints,
    water_ribbon,
)

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.layers import HeightFn

__all__ = ['RiverLayer', 'GLINT_ERROR', 'RIBBON_SPACING', 'GLINT_SPACING']

#: The tile error, in metres, past which a river stops being a surface and
#: becomes glints. Roughly where a river a few metres wide stops covering more
#: than a pixel or two: below it the shape of the water is worth drawing, above
#: it only the fact of it is.
GLINT_ERROR = 24.0

#: How far apart the surface's points are at the finest tile, in metres, and
#: how much coarser they get per metre of a tile's error. A distant tile spends
#: a fraction of the vertices on the same river.
RIBBON_SPACING = 6.0
RIBBON_PER_ERROR = 1.5

#: How far apart the glints are at the threshold, and how much further with
#: distance. They thin out as the tile coarsens, which is what keeps a river
#: seen from the far side of a world down to a handful of quads.
GLINT_SPACING = 45.0
GLINT_PER_ERROR = 3.0

#: How far above the bed the water sits, as a share of the bed's depth. A river
#: fills its channel; it does not brim over it.
SURFACE_SHARE = 0.55


@dataclass
class RiverLayer:
    """The water running down a world's channels.

    ``ground`` is the land **without** the channels in it -- what the beds were
    cut into -- because the surface is measured down from that rather than up
    from the bed, which is what puts water in a valley rather than a ribbon on
    a hillside.
    """

    channels: Sequence[Any] = field(default_factory=list)
    ground: HeightFn | None = None
    style: WaterStyle = FLOWING
    material: PBRMaterial | None = None
    name: str = 'river'
    #: Whether the card moves the surface. A baked world is streamed and
    #: redrawn every frame, so it should; a caller wanting a still picture of
    #: it turns this off.
    on_gpu: bool = True

    def bounds(self) -> BoundingBox | None:
        """The ground the rivers cover, or None where there are none."""
        return BoundingBox.joined(
            BoundingBox.of_points(np.asarray(surface.positions, dtype='d'))
            for surface in self._surfaces(spacing=None))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        """The water in one tile, at the detail that tile is drawn at."""
        glinting = float(error) >= GLINT_ERROR
        found: list[SceneNode] = []
        for index, channel in enumerate(self.channels):
            course = self._course(channel)
            if course is None or not self._crosses(course, region, channel):
                continue
            mesh = (self._glints(course, channel, error) if glinting
                    else self._ribbon(course, channel, error))
            if mesh is None:
                continue
            found.append(SceneNode(
                mesh=mesh,
                name='%s_%d%s' % (self.name, index,
                                  '_glint' if glinting else '')))
        return found

    # -- what a river is at a distance -------------------------------------
    def _ribbon(self, course: np.ndarray, channel: Any,
                error: float) -> Any:
        spacing = max(RIBBON_SPACING, float(error) * RIBBON_PER_ERROR)
        line, widths = self._resampled(course, channel, spacing)
        return water_ribbon(line, width=widths, style=self.style,
                            material=self.material, on_gpu=self.on_gpu)

    def _glints(self, course: np.ndarray, channel: Any, error: float) -> Any:
        spacing = GLINT_SPACING + max(0.0, float(error) - GLINT_ERROR) \
            * GLINT_PER_ERROR
        return water_glints(course, width=channel.widths(), spacing=spacing,
                            style=self.style, material=self.material,
                            on_gpu=self.on_gpu)

    def _resampled(self, course: np.ndarray, channel: Any,
                   spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """The course at a coarser spacing, and its width at those points."""
        steps = np.linalg.norm(np.diff(course[:, [0, 2]], axis=0), axis=1)
        along = np.concatenate([[0.0], np.cumsum(steps)])
        total = float(along[-1])
        if total <= spacing:
            return (course, channel.widths())
        at = np.linspace(0.0, total, max(2, int(round(total / spacing)) + 1))
        line = np.stack([np.interp(at, along, course[:, axis])
                         for axis in range(3)], axis=-1)
        return (line, np.interp(at, along, channel.widths()))

    # -- where the water is -------------------------------------------------
    def _course(self, channel: Any) -> np.ndarray | None:
        """The surface of one river, as (N,3) world points."""
        line = np.asarray(channel.points, dtype='d').reshape(-1, 2)
        if len(line) < 2 or self.ground is None:
            return None
        land = np.asarray(self.ground(line[:, 0], line[:, 1]), dtype='d')
        return np.stack([line[:, 0], land - channel.depths() * SURFACE_SHARE,
                         line[:, 1]], axis=-1)

    def _crosses(self, course: np.ndarray, region: BoundingBox,
                 channel: Any) -> bool:
        """Whether a river comes near enough to a tile to be in it."""
        reach = float(np.max(channel.widths())) if len(channel.points) else 0.0
        low = region.minimum - reach
        high = region.maximum + reach
        inside = ((course[:, 0] >= low[0]) & (course[:, 0] <= high[0])
                  & (course[:, 2] >= low[2]) & (course[:, 2] <= high[2]))
        return bool(inside.any())

    def _surfaces(self, spacing: float | None) -> Any:
        """Every river's surface, for measuring what they cover."""
        for channel in self.channels:
            course = self._course(channel)
            if course is None:
                continue
            mesh = water_ribbon(course, width=channel.widths(),
                                style=self.style, material=self.material)
            if mesh is not None:
                yield mesh
