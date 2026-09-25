"""Loose stone strewn over the ground, drawn in the tiles and stood up in the world.

A hillside is not a smooth surface with four boulders on it. It is made of
stone -- the hand-sized and knee-high sort, lying in the grass, too numerous to
be worth a node apiece and too small to be worth a level of detail of its own.
That is what this layer carries, and it is the other half of what a refining
tile tree should buy:
:class:`~OpenGLContext.scenegraph.terrain.Relief` puts the swells in the ground
and this puts the stone on it.

The stones ride in the tile as ``EXT_mesh_gpu_instancing`` placements -- one
node per shape, however many stones the tile holds -- and appear only at the
levels fine enough to draw them. A tile of a few hundred stones is then a few
kilobytes of placements over a shape it already carries, where the same stones
written out as geometry would be most of what the world weighs.

A stone is part of the ground, so it is solid. Walk onto one and you stand
on it; drive over one and the wheel rides over it. The stones travel as a
binary table beside the tileset (``stones.npz``,
:func:`~OpenGLContext.scenegraph.props.props_table`), which the tileset's
``extras.stones`` names with the count, and a game stands the ones near it up
with :meth:`PropColliders.baked
<OpenGLContext.physics.props.PropColliders.baked>` -- as a *dome* rather than
a boulder's box, because a block the size of a stone is a kerb
across the hillside. What separates the two is what they are for: a boulder is
an obstacle, held for hundreds of metres around because a car has to be stopped
by it; a stone is ground, and matters where a wheel is.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import InstanceSet, SceneNode
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.props import Prop, props_table

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.scatter import yaw_quaternions

#: How small a tile's geometric error has to be, against a stone's own radius,
#: before that stone is drawn: a stone appears where the error is no more than
#: this many times its radius.
#:
#: A tile is refined when its error crosses the viewer's screen-space tolerance,
#: a matter of a dozen-odd pixels, so at a tile's own error a stone a fifth that
#: size is still a few pixels across and worth having. Waiting until the error
#: is down to the stone itself holds the ground bare through the levels where a
#: player is looking straight at it.
DETAIL = 5.0

#: What the table of every stone is called beside the tileset.
TABLE = 'stones.npz'

#: The most stone one tile carries. The tree refines with REPLACE, so a tile
#: stands in for its whole subtree and a stone fine enough to draw is written
#: into every level from there down; a coarse tile is thinned to a stand-in by
#: an even stride, which bounds that and keeps a distant hillside showing stone
#: rather than showing all of it.
#:
#: This is what one tile *draws*. What a game stands up comes off the tileset's
#: own table, so a stone a thinned tile left out is still something to stand on.
MOST_PER_TILE = 256


@dataclass
class StoneLayer:
    """Small stones, drawn in the tiles that can show them and solid everywhere.

    ``stones`` are the placements, each a
    :class:`~OpenGLContext.scenegraph.props.Prop` whose ``position`` is where
    its base sits -- already bedded into the ground, so what is drawn and what
    is stood on are one number. ``prototypes`` is the mesh each ``kind`` is
    drawn as, at the origin with its base at y=0, which the placement scales
    and turns.

    ``detail`` is how soon a stone arrives, as a multiple of its radius
    (:data:`DETAIL`), and ``max_stones`` the most one tile draws
    (:data:`MOST_PER_TILE`).
    """

    stones: Sequence[Prop]
    prototypes: Mapping[str, PBRMesh]
    detail: float = DETAIL
    max_stones: int = MOST_PER_TILE
    name: str = 'stones'

    def __post_init__(self) -> None:
        self.stones = list(self.stones)
        missing = sorted({one.kind for one in self.stones}
                         - set(self.prototypes))
        if missing:
            raise ValueError(
                "no prototype to draw %s with; a stone layer needs a mesh for "
                "every kind it places" % (', '.join(missing),))
        self._plan = np.asarray([one.position for one in self.stones],
                                dtype='d').reshape(-1, 3)
        self._radii = np.asarray([one.radius for one in self.stones],
                                 dtype='d').reshape(-1)
        #: Each stone's kind, as an index into :meth:`kinds`.
        number = {kind: index for index, kind in enumerate(self.kinds())}
        self._kind = np.asarray([number[one.kind] for one in self.stones],
                                dtype='i4')
        self._scales = np.asarray([one.scale for one in self.stones],
                                  dtype='f4')
        self._yaws = np.asarray([one.yaw for one in self.stones], dtype='d')

    def kinds(self) -> list[str]:
        """The kinds this layer actually places, in a settled order."""
        return sorted({one.kind for one in self.stones})

    def bounds(self) -> BoundingBox | None:
        """The ground the stones lie on, with their own height on it."""
        if not len(self.stones):
            return None
        box = BoundingBox.of_points(self._plan)
        assert box is not None
        reach = float(self._radii.max())
        tall = max(float(one.height) for one in self.stones)
        return BoundingBox(
            (box.minimum[0] - reach, box.minimum[1], box.minimum[2] - reach),
            (box.maximum[0] + reach, box.maximum[1] + tall,
             box.maximum[2] + reach))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        mine = self._in(region, error)
        if not mine:
            return []
        held = np.asarray(mine, dtype=np.intp)
        found: list[SceneNode] = []
        for number, kind in enumerate(self.kinds()):
            wearing = held[self._kind[held] == number]
            if not len(wearing):
                continue
            size = self._scales[wearing]
            found.append(SceneNode(
                mesh=self.prototypes[kind],
                instances=InstanceSet(
                    translations=self._plan[wearing].astype('f'),
                    rotations=yaw_quaternions(self._yaws[wearing]),
                    scales=np.repeat(size[:, None], 3, axis=1)),
                name='%s-%s' % (self.name, kind)))
        return found

    def metadata(self) -> dict[str, Any]:
        """Where the table of every stone is, and how many it holds.

        Its own channel rather than the world's ``props``, because the two are
        held at different reaches: a boulder is something to be stopped by from
        a long way off, and a stone is what is under the wheel. A table rather
        than JSON, because the tileset's ``extras`` is parsed by everything
        that opens the world and a default world's stone is tens of thousands.
        """
        if not self.stones:
            return {}
        return {'stones': {'table': TABLE, 'count': len(self.stones)}}

    def assets(self) -> dict[str, bytes]:
        """The table of every stone, beside the tileset."""
        if not self.stones:
            return {}
        return {TABLE: props_table(self.stones)}

    def _in(self, region: BoundingBox, error: float) -> list[int]:
        """Which stones this tile holds and is fine enough to draw."""
        if not len(self.stones):
            return []
        held = region.holds(self._plan)
        mine = np.nonzero(held & (self._radii * self.detail >= float(error)))[0]
        if len(mine) > self.max_stones:
            # An even stride rather than a random sample: the thinned set is
            # the same every bake and stays spread over the tile.
            stride = int(np.ceil(len(mine) / self.max_stones))
            mine = mine[::stride][:self.max_stones]
        return [int(index) for index in mine]
