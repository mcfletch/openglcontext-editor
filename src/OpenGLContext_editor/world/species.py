"""The kinds of tree a world's forest is drawn from.

A :class:`~OpenGLContext.scenegraph.vegetation.field.TreeSpecies` is a set of
files -- geometry, a bark texture, a foliage texture, and the card the tree
becomes at a distance -- and a world is baked with whichever ones its author
chose. The toolkit ships none: art is the caller's, and a baker that carried a
default forest in its wheel would make every world made with it look the same.

What is here is the *shipped example* world's answer, which is the forest demo's
species set. It is used when that package is installed, and
:func:`shipped_species` says plainly what to do when it is not.

The ground *cover* between the trees comes from the same place:
:func:`shipped_cover` is the demo's grass clump and its card. The clump and the
card baked from it are the toolkit's own work and carry no attribution
requirement of their own.

**The tree assets are CC-BY 4.0** and their attributions travel with any world
baked from them: :func:`shipped_credits` returns them, and the bake writes them
into the tileset's copyright and its ``CREDITS.txt``. Baking a world with them
and shipping it without the credit is a licence breach, which is why the two are
in one module and neither is optional.
"""
from __future__ import annotations

import os
from typing import Any

from OpenGLContext.scenegraph.vegetation.cover import CoverSpecies
from OpenGLContext.scenegraph.vegetation.field import TreeSpecies

__all__ = ['shipped_species', 'shipped_credits', 'species_directory',
           'shipped_cover', 'SHIPPED', 'CREDITS', 'COVER']

#: The species the example world uses, as ``(name, files..., keys)``. Fir and
#: Noel pine are conifers, which is what a coniferous landscape wants; the two
#: maples give the valleys something broadleaf so a lap is not one tree
#: repeated.
SHIPPED = (
    dict(name='fir', mesh='fir.npz', solid_texture='fir_bark.png',
         foliage_texture='fir_branch.png', impostor='fir_imp.png',
         card_width=0.50),
    dict(name='noel', mesh='noel.npz', solid_texture='noel_bark.png',
         foliage_texture='noel_branch.png', impostor='noel_imp.png',
         card_width=0.55),
    dict(name='maple0', mesh='maple0.npz', solid_texture='maple_bark.png',
         foliage_texture='maple_leaves.png', impostor='maple_imp0.png',
         solid=('bP', 'bN', 'bU', 'bI'), foliage=('cP', 'cN', 'cU', 'cI'),
         card_width=0.72),
    dict(name='maple2', mesh='maple2.npz', solid_texture='maple_bark.png',
         foliage_texture='maple_leaves.png', impostor='maple_imp2.png',
         solid=('bP', 'bN', 'bU', 'bI'), foliage=('cP', 'cN', 'cU', 'cI'),
         card_width=0.72),
)

#: What the example world grows between its trees. A clump of real blades for
#: the near field and the card it is baked into for the far one, dense enough
#: that a driver sees ground cover rather than individual tufts.
COVER = dict(name='grass', clump='basic-clump.glb', card='grass_clump_imp.png',
             density=1.6, height=0.5)

#: What a world baked from :data:`SHIPPED` has to say about where its trees came
#: from. CC-BY 4.0 requires the attribution to travel with the work.
CREDITS = (
    "Tree models: 'Fir tree' by Georgeous (https://skfb.ly/pA8TG), "
    "'Noel_Pine_Tree' by 3D Error 404 (https://skfb.ly/6XHoJ) and "
    "'Maple trees pack' by LOLIPOP (https://skfb.ly/p9tGx), all CC-BY 4.0 "
    "(http://creativecommons.org/licenses/by/4.0/); the meshes, textures and "
    "impostor cards here are derivative works and carry the same licence.",
)


def species_directory() -> str:
    """Where the example world's tree files are.

    They are the forest demo's, which packages them; a world of your own points
    :func:`shipped_species` at its own directory instead.
    """
    try:
        import openglcontext_forest_demo
    except ImportError as error:
        raise LookupError(
            "the example world's trees come from openglcontext-forest-demo, "
            "which is not installed. Install it, pass a directory of your own "
            "to shipped_species(), or bake the world with trees='tiles'."
        ) from error
    return os.path.join(os.path.dirname(openglcontext_forest_demo.__file__),
                        'assets')


def shipped_species(directory: str | None = None) -> list[TreeSpecies]:
    """The example world's species, with their files resolved.

    ``directory`` holds the files; it defaults to
    :func:`species_directory`. Every named file has to be there, because a
    species missing its bark is a tree that fails to draw at the moment a player
    walks up to it rather than at the moment the world is baked.
    """
    where = directory or species_directory()
    found = []
    for entry in SHIPPED:
        species = TreeSpecies(**entry).beside(where)
        for part in (species.mesh, species.solid_texture,
                     species.foliage_texture, species.impostor):
            if not os.path.exists(part):
                raise LookupError(
                    "%s is part of the '%s' tree and is not in %s"
                    % (os.path.basename(part), species.name, where))
        found.append(species)
    return found


def shipped_cover(directory: str | None = None) -> CoverSpecies:
    """The example world's ground cover, with its files resolved.

    ``directory`` holds the files; it defaults to :func:`species_directory`.
    """
    where = directory or species_directory()
    cover = CoverSpecies(**COVER).beside(where)
    for part in (cover.card, cover.clump):
        if part and not os.path.exists(part):
            raise LookupError(
                "%s is part of the '%s' ground cover and is not in %s"
                % (os.path.basename(part), cover.name, where))
    return cover


def shipped_credits() -> tuple[str, ...]:
    """The attributions a world baked from :func:`shipped_species` must carry."""
    return CREDITS


def species_are_available(directory: str | None = None) -> bool:
    """Whether :func:`shipped_species` would find its files.

    For a caller choosing what to bake rather than one baking it: a world that
    would like a forest but can run without one.
    """
    try:
        shipped_species(directory)
    except LookupError:
        return False
    return True


def biome_species(positions: Any, heights: Any, slopes: Any, seed: int = 11
                  ) -> Any:
    """Which of the shipped species stands at each place.

    Conifers take the high and the steep ground, maples the low and gentle, with
    a broad noise over the top so a hillside has a dominant kind and a minority
    mixed through it rather than every kind everywhere.
    """
    import numpy as np
    if not len(np.asarray(positions, dtype='d')):
        return np.zeros(0, 'i4')
    x = np.asarray(positions, dtype='d')[:, 0]
    y = np.asarray(positions, dtype='d')[:, 1]
    z = np.asarray(positions, dtype='d')[:, 2]
    steep = np.clip(np.asarray(slopes, dtype='d'), 0.0, 1.5)
    high = np.clip((y - y.min()) / max(float(y.max() - y.min()), 1e-6), 0.0, 1.0)
    # A long-wavelength field, so the mix changes over hundreds of metres rather
    # than tree to tree.
    patch = np.sin(x * 0.004 + 1.3) * np.cos(z * 0.0035) \
        + 0.5 * np.sin(z * 0.010)
    patch = (patch - patch.min()) / max(float(patch.max() - patch.min()), 1e-6)
    rng = np.random.default_rng(seed)
    conifer = rng.random(len(x)) < np.clip(
        high * 1.4 + steep * 1.2 + (patch - 0.5) * 1.1 - 0.28, 0.03, 0.97)
    kind = np.where(conifer, rng.integers(0, 2, len(x)),
                    2 + rng.integers(0, 2, len(x)))
    return kind.astype('i4')
