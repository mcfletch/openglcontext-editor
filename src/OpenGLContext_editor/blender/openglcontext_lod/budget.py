"""How many triangles each level of a chain is allowed, and what to ask for.

An author thinks in budgets: *the finest level is not to exceed twenty
thousand, and each one after that is half the one before*. Blender's Decimate
modifier thinks in ratios -- ``ratio`` is the only target it takes, and its
``face_count`` is read-only and merely reports what came out. This turns the
first into the second.

Every ratio is measured against the **original** mesh, because that is what the
modifier does: one modifier on the source object, asked for a different ratio
per level. Asking for a half twice is a quarter of the source, not a quarter of
the level before.

No Blender in here.
"""

from __future__ import annotations

__all__ = ['FLOOR', 'describe', 'ratios_for', 'targets_for']

#: The fewest triangles a level is allowed. Below about this a reduction stops
#: being a coarser model and becomes a fold: a closed surface needs four faces,
#: and a silhouette worth drawing needs more than that.
FLOOR = 8


def targets_for(source_triangles: int, levels: int, ratio: float = 0.5,
                max_triangles: int | None = None) -> list[int]:
    """How many triangles each of ``levels`` levels should end up with.

    The finest is the mesh as it came, or ``max_triangles`` where it arrived
    denser than that -- a model at whatever density its author left it is not a
    budget, and a chain whose first rung is half a million triangles has not
    begun to help. Each level after it keeps ``ratio`` of the one before, and
    none falls below :data:`FLOOR`.
    """
    _refuse(source_triangles, levels, ratio)
    finest = int(source_triangles)
    if max_triangles:
        finest = min(finest, int(max_triangles))
    finest = max(finest, FLOOR if source_triangles >= FLOOR else source_triangles)
    made = [finest]
    for _level in range(1, levels):
        nearer = made[-1]
        made.append(max(FLOOR, int(round(nearer * ratio))) if nearer > FLOOR
                    else nearer)
    return made


def ratios_for(source_triangles: int, levels: int, ratio: float = 0.5,
               max_triangles: int | None = None) -> list[float]:
    """What to set the modifier's ``ratio`` to for each level.

    One per level, against the original, never above one: a level is a
    reduction of the mesh or it is the mesh.
    """
    source = max(1, int(source_triangles))
    return [min(1.0, max(1.0 / source, target / source))
            for target in targets_for(source_triangles, levels, ratio,
                                      max_triangles)]


def describe(targets: list[int]) -> str:
    """One line saying what a chain will cost, for a report to the author."""
    if not targets:
        return 'no levels'
    if len(targets) == 1:
        return 'one level, %s triangles' % (f'{targets[0]:,}',)
    return '%d levels, %s down to %s triangles' % (
        len(targets), f'{targets[0]:,}', f'{targets[-1]:,}')


def _refuse(source_triangles: int, levels: int, ratio: float) -> None:
    if levels < 1:
        raise ValueError('a chain needs at least one level')
    if source_triangles < 1:
        raise ValueError('a chain needs a mesh with triangles in it')
    if not 0.0 < ratio < 1.0:
        raise ValueError(
            'a level keeps between none and all of the triangles before it; '
            '%r would not coarsen the mesh' % (ratio,))
