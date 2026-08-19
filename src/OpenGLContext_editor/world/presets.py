"""Landscapes to start from.

A designer opening a track editor wants ground worth drawing a road across, and
"the shipped landscape" is one answer to a question with several. Each preset
here is a tuned :class:`~OpenGLContext.loaders.tiles3d.procedural.TerrainProfile`
-- how much hill, how much mountain, how deep a canyon, how broad a basin --
under a name and a sentence saying what it is for.

They are data rather than code, so a game editor built on this toolkit gets the
same starting points, and a project that names one reproduces the same ground
when it is opened tomorrow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from OpenGLContext.loaders.tiles3d.procedural import (
    SHIPPED_TERRAIN,
    TerrainProfile,
    terrain_height_for,
)

from OpenGLContext_editor.world.height import HeightBase, register_base

__all__ = ['PRESETS', 'Preset', 'PresetBase']


@dataclass(frozen=True)
class Preset:
    """One named landscape: what to call it, and what it is made of."""

    #: What a menu calls it.
    label: str
    #: One sentence: what a designer gets, and what it is good for.
    description: str
    profile: TerrainProfile


#: The landscapes on offer. The numbers are metres of relief and metres on the
#: ground, so what each one is can be read off the entry.
PRESETS: dict[str, Preset] = {
    'shipped': Preset(
        label='The shipped landscape',
        description="Hill country with a range along one side, a river canyon "
                    "and a lake basin. What every worked example is built on.",
        profile=SHIPPED_TERRAIN),
    'mountains': Preset(
        label='Dramatic mountains',
        description="Ranges over most of the map, rising most of a kilometre. "
                    "A road across it climbs, tunnels or is carried.",
        profile=TerrainProfile(hills=70.0, mountains=900.0,
                               mountain_scale=1400.0, mountain_cover=0.82,
                               canyon=0.0, basin=0.0, datum=60.0)),
    'lakes': Preset(
        label='Lakes',
        description="Low country dished out into broad basins that flood, with "
                    "gentle ground between them. Causeways and bridges.",
        profile=TerrainProfile(hills=55.0, hill_scale=900.0, mountains=90.0,
                               mountain_scale=1600.0, mountain_cover=0.35,
                               canyon=0.0, basin=190.0, basin_scale=1500.0,
                               datum=26.0)),
    'hills': Preset(
        label='Rolling hills',
        description="Nothing steeper than a road can climb, anywhere. The "
                    "landscape to draw a fast circuit on.",
        profile=TerrainProfile(hills=40.0, hill_scale=1100.0, mountains=0.0,
                               canyon=0.0, basin=0.0, datum=30.0)),
    'canyon': Preset(
        label='Canyon',
        description="A gorge three hundred metres deep winding across a high "
                    "plain. One crossing decides the whole circuit.",
        profile=TerrainProfile(hills=35.0, hill_scale=800.0, mountains=60.0,
                               mountain_scale=1200.0, mountain_cover=0.30,
                               canyon=320.0, canyon_width=140.0,
                               canyon_meander=420.0, basin=0.0, datum=150.0)),
}


@register_base
@dataclass(frozen=True)
class PresetBase(HeightBase):
    """One of the shipped landscapes, at a relief and a seed.

    ``relief`` multiplies its height, which is how a landscape too tall for a
    road is made into one a road can be built through without changing what it
    looks like. ``seed`` gives another landscape of the same description --
    another set of ranges, another course for the river -- rather than another
    kind of landscape.
    """

    KIND: ClassVar[str] = 'preset'
    name: str = 'shipped'
    relief: float = 1.0
    seed: int = 0

    def preset(self) -> Preset:
        """The entry this base names."""
        if self.name not in PRESETS:
            raise ValueError(
                "this landscape is %r, which this version does not ship" %
                (self.name,))
        return PRESETS[self.name]

    def profile(self) -> TerrainProfile:
        """The terrain profile, with this base's seed on it."""
        from dataclasses import replace
        return replace(self.preset().profile, seed=int(self.seed))

    def sample(self, x: Any, z: Any) -> np.ndarray:
        ground = np.asarray(terrain_height_for(self.profile())(x, z), dtype='d')
        if self.relief == 1.0:
            return ground
        return ground * float(self.relief)

    def to_json(self) -> dict[str, Any]:
        return {'kind': self.KIND, 'name': self.name,
                'relief': float(self.relief), 'seed': int(self.seed)}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> PresetBase:
        return cls(name=str(document.get('name', 'shipped')),
                   relief=float(document.get('relief', 1.0)),
                   seed=int(document.get('seed', 0)))
