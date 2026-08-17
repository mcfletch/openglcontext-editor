"""Ground baked as one field rather than as a tree of tiles.

A tiled ground gets finer as the tree refines, which is what a world larger than
memory needs. A world of a few kilometres does not need it: the whole landscape
fits in one
:class:`~OpenGLContext.scenegraph.terrain.heightfield.HeightField`, and rendering
it as a :class:`~OpenGLContext.scenegraph.terrain.splat.SplatTerrain` -- one
mesh, one draw, detail materials blended per pixel -- costs less than a tree of
vertex-coloured patches and carries crisp ground right up to the camera.

So a world chooses. :class:`FieldTerrainLayer` is the field choice: it puts no
geometry in any tile, and writes the landscape beside the tileset as a 16-bit
height image and an RGBA splat control map, with the numbers to read them back
in the tileset's ``extras``. What still comes from tiles is everything that is
not ground -- the road, its structures, anything placed.

A game that meets a world with ``extras.terrain`` builds the field from those
two images and renders and collides against it; see
:mod:`OpenGLContext.physics.heightfield` for the collider side.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import SceneNode
from OpenGLContext.scenegraph.terrain import HeightField, LayerRule, control_map

from OpenGLContext_editor.bake.bounds import BoundingBox

HeightFn = Callable[[Any, Any], Any]

#: How many samples across the landscape's height grid. 513 over four kilometres
#: is eight-metre spacing, which is the forest demo's own figure and the point
#: at which a splat terrain's detail materials, not its silhouette, are what a
#: player is looking at.
DEFAULT_RESOLUTION = 513

#: How many samples across the splat control map. Where the ground changes
#: material is a broader thing than where it changes height, so it need not
#: match.
DEFAULT_CONTROL_SIZE = 512

#: The ground materials a landscape is made of, and where each belongs. Named
#: for :mod:`OpenGLContext.loaders.cc0`, which resolves them to CC0 texture
#: sets. The first is the fallback: ground no rule wants is made of it.
DEFAULT_LAYERS = ('grass', 'forest_floor', 'rock', 'dirt')
DEFAULT_RULES = (
    LayerRule(),
    LayerRule(slope=(0.18, 0.55), weight=1.4),
    LayerRule(slope=(0.5, 1.0e9), weight=2.5),
    LayerRule(weight=0.0),
)

#: How far either side of a road's centreline the ground takes the road's own
#: material, as a multiple of the road's total width, when a caller does not say
#: in metres. Wider than the carriageway, because what is being painted is the
#: disturbed ground a road is built in.
ROAD_CORRIDOR = 1.35


@dataclass
class FieldTerrainLayer:
    """The landscape as one height field and one splat control map.

    ``height_fn`` is the ground -- including any earthworks a road has cut into
    it -- and ``extent`` the footprint it covers; the footprint's Y is ignored,
    since the height function decides that.

    ``height_fn_at(spacing) -> height_fn`` is the same ground as a function of
    how far apart it will be sampled, and is what a world with a road in it
    should give instead. A cutting narrower than the grid is stepped straight
    over, however deep the height function says it is at its centre, and the
    road then comes out buried in the hillside it was cut into; a height
    function told the spacing holds a shelf that wide at the verge, so a sample
    lands inside the corridor whatever the resolution. See
    :func:`~OpenGLContext_editor.world.road.conform_terrain_at`.

    ``layers`` names the detail materials, up to four, and ``rules`` says where
    each belongs (see
    :class:`~OpenGLContext.scenegraph.terrain.control.LayerRule`). ``road``, if
    given, paints its corridor into ``road_layer`` so the ground beside the
    carriageway is that material rather than whatever the slope suggested;
    ``road_corridor`` is how far out that reaches from the centreline, in metres,
    and should end about where whatever was cleared for the road ends.
    """

    height_fn: HeightFn
    extent: BoundingBox
    height_fn_at: Any = None
    resolution: int = DEFAULT_RESOLUTION
    control_size: int = DEFAULT_CONTROL_SIZE
    layers: Sequence[str] = DEFAULT_LAYERS
    rules: Sequence[LayerRule] = DEFAULT_RULES
    road: Any = None
    road_layer: int = 3
    road_corridor: float | None = None
    name: str = 'terrain'
    _field: HeightField | None = dataclass_field(default=None, init=False,
                                                 repr=False)

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.rules):
            raise ValueError(
                "%d ground materials need %d rules, not %d"
                % (len(self.layers), len(self.layers), len(self.rules)))

    @property
    def side(self) -> float:
        """How wide the landscape is, in metres."""
        return float(self.extent.maximum[0] - self.extent.minimum[0])

    def field(self) -> HeightField:
        """The landscape, sampled from the height function once and kept.

        Sampling a conformed height function over a quarter of a million points
        is not something to do twice.
        """
        if self._field is None:
            spacing = self.side / max(self.resolution - 1, 1)
            ground = (self.height_fn_at(spacing) if self.height_fn_at is not None
                      else self.height_fn)
            self._field = HeightField.from_function(
                ground, res=self.resolution, extent=self.side)
        return self._field

    def bounds(self) -> BoundingBox:
        """The ground, with the height the field found in it."""
        low, high = self.extent.minimum, self.extent.maximum
        ground = self.field()
        return BoundingBox((low[0], ground.base, low[2]),
                           (high[0], ground.base + ground.relief, high[2]))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        """Nothing: the ground is not tiled, it is the field beside the tileset."""
        return []

    def metadata(self) -> dict[str, Any]:
        """The numbers a game needs to read the two images back.

        Without them the height image is a grey square: it says how the ground
        rises and nothing about how far, how wide, or where its zero stands.
        """
        ground = self.field()
        return {'terrain': {
            'height': self._filename('height'),
            'control': self._filename('control'),
            'extent': round(self.side, 4),
            'base': round(ground.base, 4),
            'relief': round(ground.relief, 4),
            'resolution': int(self.resolution),
            'layers': list(self.layers),
        }}

    def assets(self) -> dict[str, bytes]:
        """The height image and the control map, written beside the tileset."""
        ground = self.field()
        height = io.BytesIO()
        ground.save_image(height)
        control = io.BytesIO()
        control_map(ground, list(self.rules), size=self.control_size,
                    painted=self._painted(ground)).save(control, format='PNG')
        return {self._filename('height'): height.getvalue(),
                self._filename('control'): control.getvalue()}

    def _filename(self, part: str) -> str:
        return '%s-%s.png' % (self.name, part)

    def _painted(self, ground: HeightField) -> list[tuple[int, np.ndarray]]:
        """Masks the rules could not have worked out: so far, the road's."""
        if self.road is None:
            return []
        size = self.control_size
        axis = np.linspace(-self.side / 2.0, self.side / 2.0, size)
        x, z = np.meshgrid(axis, axis)
        corridor = (self.road_corridor if self.road_corridor is not None
                    else self.road.profile.total_width * ROAD_CORRIDOR)
        found = self.road.sample(x, z, radius=corridor * 1.6)
        # Full strength on the road, fading out over the last quarter of the
        # corridor, so the material does not end in a line across the ground.
        fade = corridor * 0.25
        cover = np.clip((corridor - found.distance.reshape(x.shape))
                        / max(fade, 1e-6), 0.0, 1.0)
        # Only where the road is laid on the land. A deck stands over the
        # ground and a bore runs inside it; painting a corridor of bare earth
        # under either would draw the road's path across an untouched hillside.
        cover *= self.road.segment_on_ground[found.segment.reshape(x.shape)]
        return [(self.road_layer, cover)]
