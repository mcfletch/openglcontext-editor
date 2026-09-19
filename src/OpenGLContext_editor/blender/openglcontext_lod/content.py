"""What the gallery is made of, and where those files are.

The art is CC0 and is fetched rather than committed, so the add-on is told a
directory and works out the rest. What each surface is made of is a decision
about the demo rather than about Blender, so it is written down here as data:

======================  ==============================================
``GalleryFloor``        ambientCG ``WoodFloor070`` -- dark parquet, with
                        a clearcoat over it for the polish
``GalleryWall``         ambientCG ``PaintedPlaster017``
``GalleryCeiling``      ambientCG ``PaintedPlaster017``, between the beams
``GalleryBeam``         ambientCG ``Wood067`` -- dark, near-black oak
``GalleryPlinth``       ambientCG ``Plaster001``, at a tighter tile
======================  ==============================================

Everything named here is CC0 from `ambientCG <https://ambientcg.com>`_, and the
bust is CC0 from `Poly Haven <https://polyhaven.com>`_. Nothing in this module
imports Blender.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    'MATERIALS',
    'MaterialMaps',
    'Recipe',
    'BUST',
    'bust_path',
    'gallery_content',
    'material_maps',
]

#: The Poly Haven asset the demo is a gallery of.
BUST = 'marble_bust_01'


@dataclass
class MaterialMaps:
    """The image files and constants one Blender material is built from.

    ``colour`` is read as sRGB and the rest as data: a roughness map is a
    number per texel, and reading it through a transfer curve would bend every
    one of them.
    """

    colour: str | None = None
    roughness: str | None = None
    normal: str | None = None
    base_colour: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness_value: float = 0.5
    metallic: float = 0.0
    #: How much clearcoat sits over the surface, as
    #: ``KHR_materials_clearcoat`` on the way out.
    coat: float = 0.0
    coat_roughness: float = 0.03
    #: How many metres apart the texture repeats.
    tile: float = 2.0


@dataclass(frozen=True)
class Recipe:
    """One surface: which CC0 material it is, and how it is finished."""

    asset: str
    tile: float = 2.0
    coat: float = 0.0
    coat_roughness: float = 0.03
    #: Used where the asset has no colour map of its own.
    base_colour: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness_value: float = 0.5
    #: Which maps to use, of those the asset has.
    maps: tuple[str, ...] = ('color', 'roughness', 'normal')


#: A polished floor is a rough-ish wood under a smooth lacquer, which is what a
#: clearcoat is for: the coat reflects the room sharply while the grain under it
#: stays matt. One layer, rather than a floor pretending to be a mirror.
MATERIALS: dict[str, Recipe] = {
    'GalleryFloor': Recipe('WoodFloor070', tile=2.0, coat=0.9,
                           coat_roughness=0.035),
    'GalleryWall': Recipe('PaintedPlaster017', tile=3.0),
    'GalleryCeiling': Recipe('PaintedPlaster017', tile=3.0),
    'GalleryBeam': Recipe('Wood067', tile=1.5),
    'GalleryPlinth': Recipe('Plaster001', tile=1.0),
}

#: What a map is called once it is in the content directory.
_FILES = {'color': 'color.jpg', 'roughness': 'roughness.jpg',
          'normal': 'normal.jpg'}


def bust_path(content: str, asset: str = BUST) -> str:
    """The bust's glTF inside a content directory."""
    for leaf in ('%s_1k.gltf' % (asset,), '%s.gltf' % (asset,),
                 '%s.glb' % (asset,)):
        candidate = os.path.join(content, asset, leaf)
        if os.path.exists(candidate):
            return candidate
    raise OSError('no %s model under %s' % (asset, content))


def material_maps(content: str, recipe: Recipe) -> MaterialMaps:
    """The files ``recipe`` names, under ``content``.

    A map the asset does not carry is left out rather than faked; the material
    then uses its constant for that channel.
    """
    directory = os.path.join(content, 'materials', recipe.asset)
    found: dict[str, str] = {}
    for kind in recipe.maps:
        path = os.path.join(directory, _FILES[kind])
        if os.path.exists(path):
            found['colour' if kind == 'color' else kind] = path
    if not found:
        raise OSError('no maps for %s under %s' % (recipe.asset, directory))
    return MaterialMaps(colour=found.get('colour'),
                        roughness=found.get('roughness'),
                        normal=found.get('normal'),
                        base_colour=recipe.base_colour,
                        roughness_value=recipe.roughness_value,
                        coat=recipe.coat, coat_roughness=recipe.coat_roughness,
                        tile=recipe.tile)


def gallery_content(content: str) -> tuple[str, dict[str, MaterialMaps]]:
    """``(bust, materials)`` for :func:`scene.build_gallery`."""
    return (bust_path(content),
            {name: material_maps(content, recipe)
             for name, recipe in MATERIALS.items()})
