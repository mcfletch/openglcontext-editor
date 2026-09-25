"""What a material says about being an octahedral impostor.

One place, because three of them read it: the baker writes it onto the Blender
material, the glTF export hook copies it into the material's ``extras``, and
OpenGLContext reads it back out of there. No Blender and no glTF in here -- it
is the names and the rules, so the three cannot disagree about either.
"""

from __future__ import annotations

from typing import Any

__all__ = ['ENGINES', 'HEMI', 'PROPERTY_HEMI', 'PROPERTY_VIEWS', 'VIEWS', 'mark',
           'of', 'render_engine']

#: What the glTF ``extras`` call them.
VIEWS = 'octahedralViews'
HEMI = 'octahedralHemi'

#: What an impostor's views are rendered with, best first. EEVEE reads the
#: model's own materials and is quick; it is ``BLENDER_EEVEE_NEXT`` in Blender
#: 4.2 to 4.x and ``BLENDER_EEVEE`` again from 5.0. Workbench ignores the
#: materials and is the last resort; Cycles is better than either and far
#: slower than both for a picture this small.
ENGINES = ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH')

#: What the Blender material carries, in Blender's own naming.
PROPERTY_VIEWS = 'octahedral_views'
PROPERTY_HEMI = 'octahedral_hemi'


def mark(holder: Any, views: int, hemi: bool = True) -> None:
    """Say that ``holder`` -- a Blender material -- is an impostor's."""
    holder[PROPERTY_VIEWS] = int(views)
    holder[PROPERTY_HEMI] = bool(hemi)


def of(holder: Any) -> dict | None:
    """What ``holder`` says, as the ``extras`` to write, or None if it says
    nothing.

    A grid of fewer than two views a side is not an atlas -- it is one picture,
    which is an ordinary textured card and wants none of this.
    """
    views = holder.get(PROPERTY_VIEWS)
    if views is None:
        return None
    try:
        views = int(views)
    except (TypeError, ValueError):
        return None
    if views < 2:
        return None
    return {VIEWS: views, HEMI: bool(holder.get(PROPERTY_HEMI, True))}


def render_engine(offered: Any) -> str:
    """The first of :data:`ENGINES` the running Blender ``offered``."""
    names = set(offered)
    for engine in ENGINES:
        if engine in names:
            return engine
    raise LookupError('Blender offers none of %s' % (', '.join(ENGINES),))
