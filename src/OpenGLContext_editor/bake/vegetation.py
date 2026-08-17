"""A forest baked as a table beside the tileset rather than into its tiles.

Trees written into tiles arrive and leave with the tile they stand in, which
puts the level of detail in the tile's hands. A forest does not work that way:
what a tree is drawn as depends on how far it is from the *camera*, and the
tree a hundred metres ahead is the same tree whichever tile it happens to be
over. Baked per tile it also carries its bark and its leaves in every copy of
every tile it appears in.

So :class:`VegetationLayer` writes the forest once: a table of positions,
yaws, heights and species beside the tileset, the species' own files copied in
next to it, and a record in the tileset's ``extras`` naming them. A viewer
meeting that builds a
:class:`~OpenGLContext.scenegraph.vegetation.field.VegetationField`, which draws
real geometry near the camera and cards beyond it and re-chooses both as the
camera moves.

The scatter is still decided *here*, at bake time -- where the trees stand, how
tall they are and which kind each is are decisions about the world, made once
with the road's corridor kept clear, not something a runtime should be
re-rolling.

The ground *cover* between them goes the other way. There is far too much ground
to write a blade of grass for every square metre of it, and none of those blades
is a decision anybody made, so what travels is the recipe -- a clump, a card, how
dense, and which of the splat map's layers it grows on -- and the runtime
scatters it around the camera. See
:class:`~OpenGLContext.scenegraph.vegetation.cover.GroundCover`.
"""
from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from OpenGLContext.loaders.gltf.writer import SceneNode
from OpenGLContext.scenegraph.vegetation.cover import CoverSpecies
from OpenGLContext.scenegraph.vegetation.field import TreeSpecies

from OpenGLContext_editor.bake.bounds import BoundingBox

#: Where a world keeps the files its trees are drawn from, relative to the
#: tileset. A directory rather than the tileset's own, because a species is
#: half a dozen files and a world is one.
SPECIES_DIRECTORY = 'trees'

#: Which of the ground's splat layers cover grows on when the caller does not
#: say. Grass and leaf litter are the soft ground; rock and dirt are not, and
#: the road's corridor is painted out of all of them before the map is written.
COVER_ON = ('grass', 'forest_floor')


@dataclass
class VegetationLayer:
    """Every tree in a world, written once beside the tileset.

    ``positions`` are the (N,3) trunk bases, ``yaws`` the rotation about the
    vertical in radians, and ``heights`` how tall each tree is in metres --
    which is also its instance scale. ``species_id`` says which of ``species``
    each tree is; left out, they are dealt round-robin.

    Each :class:`~OpenGLContext.scenegraph.vegetation.field.TreeSpecies` names
    its files on the machine doing the baking; they are copied into the world,
    and the record written into the tileset names the copies. A baked world is
    self-contained: it does not refer to paths on whatever machine made it.
    """

    positions: Any
    heights: Any
    species: Sequence[TreeSpecies]
    yaws: Any = None
    species_id: Any = None
    cover: CoverSpecies | None = None
    cover_on: Sequence[str] | None = None
    name: str = 'trees'

    def __post_init__(self) -> None:
        if not self.species:
            raise ValueError("a vegetation layer needs at least one species")
        self.positions = np.asarray(self.positions, dtype='f4').reshape(-1, 3)
        self.heights = np.asarray(self.heights, dtype='f4').reshape(-1)
        count = len(self.positions)
        self.yaws = (np.zeros(count, 'f4') if self.yaws is None
                     else np.asarray(self.yaws, dtype='f4').reshape(-1))
        self.species_id = (np.arange(count) % len(self.species)
                           if self.species_id is None
                           else np.asarray(self.species_id, dtype='i4').reshape(-1))
        if not count == len(self.yaws) == len(self.heights) == len(self.species_id):
            raise ValueError(
                "a forest of %d trees needs %d yaws, heights and species, not "
                "%d, %d and %d" % (count, count, len(self.yaws),
                                   len(self.heights), len(self.species_id)))

    @property
    def tree_count(self) -> int:
        return len(self.positions)

    def bounds(self) -> BoundingBox | None:
        """Everything the forest stands in, with the trees' own height on it.

        A bake's region has to hold the tops of the trees, not only the ground
        they are rooted in, or a viewer culls a hillside of them by their
        footprint.
        """
        box = BoundingBox.of_points(self.positions)
        if box is None:
            return None
        tallest = float(self.heights.max()) if len(self.heights) else 0.0
        return BoundingBox(box.minimum,
                           (box.maximum[0], box.maximum[1] + tallest,
                            box.maximum[2]))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        """Nothing: the forest is a table beside the tileset, not tile content."""
        return []

    def metadata(self) -> dict[str, Any]:
        """Where the table is and what the trees in it are drawn from."""
        record: dict[str, Any] = {
            'trees': self._table_name(),
            'count': self.tree_count,
            'species': [self._written(entry).to_json() for entry in self.species],
        }
        if self.cover is not None:
            grown = self.cover.to_json()
            grown['card'] = self._under(self.cover.card)
            grown['clump'] = (self._under(self.cover.clump)
                              if self.cover.clump else None)
            grown['on'] = list(self.cover_on if self.cover_on is not None
                               else COVER_ON)
            record['cover'] = grown
        return {'vegetation': record}

    def assets(self) -> dict[str, bytes]:
        """The table, and every file the species are drawn from.

        Files are read from wherever the species named them and written under
        the world's own tree directory. A species named twice -- two kinds
        sharing one bark texture -- is written once.
        """
        written: dict[str, bytes] = {self._table_name(): self._table()}
        sources = [source for entry in self.species
                   for source in (entry.mesh, entry.solid_texture,
                                  entry.foliage_texture, entry.impostor)]
        if self.cover is not None:
            sources.extend(part for part in (self.cover.card, self.cover.clump)
                           if part)
        for source in sources:
            name = self._under(source)
            if name not in written:
                with open(source, 'rb') as handle:
                    written[name] = handle.read()
        return written

    def _table_name(self) -> str:
        return '%s.npz' % (self.name,)

    def _table(self) -> bytes:
        """The forest as one compressed array file.

        A quarter of a million trees is four megabytes of binary and eighty of
        JSON, and the tileset's ``extras`` is read by everything that opens the
        world.
        """
        buffer = io.BytesIO()
        np.savez_compressed(buffer, positions=self.positions, yaws=self.yaws,
                            heights=self.heights, species=self.species_id)
        return buffer.getvalue()

    def _written(self, entry: TreeSpecies) -> TreeSpecies:
        """A species as the world carries it: its own copies, under the world."""
        return replace(
            entry,
            mesh=self._under(entry.mesh),
            solid_texture=self._under(entry.solid_texture),
            foliage_texture=self._under(entry.foliage_texture),
            impostor=self._under(entry.impostor))

    def _under(self, source: str) -> str:
        """Where a species' file lands, relative to the tileset."""
        return '%s/%s' % (SPECIES_DIRECTORY, os.path.basename(source))
