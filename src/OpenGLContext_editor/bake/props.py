"""A world's obstacles, written into the tiles and into the tileset.

The geometry rides in the tiles like everything else: one instanced node per
kind, so a hundred boulders are one draw. The *bodies* cannot ride with it. Tile
geometry is level-of-detail geometry that arrives and leaves as a camera moves,
and a collider built from it would be a rock a car drives through at the moment
the tile behind it swaps -- so the props travel in the tileset's ``extras`` as
well, the same way the road does, and a game stands them up itself with
:class:`OpenGLContext.physics.props.PropColliders`.

What a prop looks like is the caller's: ``prototypes`` maps a kind to the mesh
it is drawn as. The toolkit generates one kind, because a landscape supplies it
for free -- see :func:`OpenGLContext.scenegraph.props.rock_mesh`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import InstanceSet, SceneNode
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.props import Prop

from OpenGLContext_editor.bake.bounds import BoundingBox

#: The coarsest tile that carries props. A boulder is a couple of metres; a tile
#: whose error is tens of metres cannot show it and should not spend bandwidth
#: on it. Its *body* is unaffected -- that comes off the tileset, not the tile.
MAXIMUM_ERROR = 18.0


@dataclass
class PropLayer:
    """Every placed obstacle in a world.

    ``props`` are where they stand; ``prototypes`` is the mesh each kind is
    drawn as, at the origin with its base at y=0, which the placement scales
    and turns.
    """

    props: Sequence[Prop]
    prototypes: Mapping[str, PBRMesh]
    maximum_error: float = MAXIMUM_ERROR
    name: str = 'props'

    def __post_init__(self) -> None:
        missing = sorted({one.kind for one in self.props}
                         - set(self.prototypes))
        if missing:
            raise ValueError(
                "no prototype to draw %s with; a prop layer needs a mesh for "
                "every kind it places" % (', '.join(missing),))

    def kinds(self) -> list[str]:
        """The kinds this world actually places, in a settled order."""
        return sorted({one.kind for one in self.props})

    def bounds(self) -> BoundingBox | None:
        """The ground the props stand on, with their own height on it."""
        if not self.props:
            return None
        feet = np.asarray([one.position for one in self.props], dtype='d')
        box = BoundingBox.of_points(feet)
        assert box is not None
        reach = max(float(one.radius) for one in self.props)
        top = max(float(one.height) for one in self.props)
        return BoundingBox(
            (box.minimum[0] - reach, box.minimum[1], box.minimum[2] - reach),
            (box.maximum[0] + reach, box.maximum[1] + top,
             box.maximum[2] + reach))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        if error > self.maximum_error:
            return []
        found: list[SceneNode] = []
        for kind in self.kinds():
            mine = [one for one in self.props
                    if one.kind == kind and _inside(one.position, region)]
            if not mine:
                continue
            found.append(SceneNode(
                mesh=self.prototypes[kind],
                instances=InstanceSet(
                    translations=np.asarray([one.position for one in mine], 'f'),
                    rotations=_yaws([one.yaw for one in mine]),
                    scales=np.repeat(
                        np.asarray([[one.scale] for one in mine], 'f'), 3,
                        axis=1)),
                name='%s-%s' % (self.name, kind)))
        return found

    def metadata(self) -> dict[str, Any]:
        """Every prop, for the game that has to collide with them.

        A pile of triangles in a tile does not say where a boulder is, and the
        tile it is in comes and goes. See
        :class:`OpenGLContext.physics.props.PropColliders`.
        """
        return {'props': [one.to_json() for one in self.props]}


def _inside(position: Any, region: BoundingBox) -> bool:
    at = np.asarray(position, dtype='d')
    return bool(np.all(at >= region.minimum) and np.all(at <= region.maximum))


def _yaws(angles: Sequence[float]) -> np.ndarray:
    """Rotations about the vertical, as the quaternions glTF instancing wants."""
    half = np.asarray(angles, dtype='d') / 2.0
    return np.stack([np.zeros(len(half)), np.sin(half), np.zeros(len(half)),
                     np.cos(half)], axis=-1).astype('f')
