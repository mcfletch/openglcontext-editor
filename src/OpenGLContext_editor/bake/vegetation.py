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
from dataclasses import dataclass
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
    cover: CoverSpecies | Sequence[CoverSpecies] | None = None
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
        grown = self._cover_species()
        if grown:
            record['cover'] = {
                'on': list(self.cover_on if self.cover_on is not None
                           else COVER_ON),
                'species': [self._grown(entry) for entry in grown],
            }
        return {'vegetation': record}

    def _cover_species(self) -> list[CoverSpecies]:
        """What grows between the trees, as a list however it was given.

        A world with one kind of cover should not have to say so twice, so one
        species on its own is a set of one.
        """
        if self.cover is None:
            return []
        if isinstance(self.cover, CoverSpecies):
            return [self.cover]
        return list(self.cover)

    def _grown(self, entry: CoverSpecies) -> dict:
        """A cover species as the world carries it: its own copies, under it."""
        record = entry.to_json()
        record['card'] = self._under(entry.card) if entry.card else ''
        record['clump'] = self._under(entry.clump) if entry.clump else None
        return record

    def assets(self) -> dict[str, bytes]:
        """The table, and every file the species are drawn from.

        Files are read from wherever the species named them and written under
        the world's own tree directory. A species named twice -- two kinds
        sharing one bark texture -- is written once.
        """
        written: dict[str, bytes] = {self._table_name(): self._table()}
        for source, name in self._placed().items():
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
        return entry.varied(
            mesh=self._under(entry.mesh),
            solidTexture=self._under(entry.solidTexture),
            foliageTexture=self._under(entry.foliageTexture),
            impostor=self._under(entry.impostor))

    def _under(self, source: str) -> str:
        """Where a species' file lands, relative to the tileset.

        Under its own name, and where two files from different directories
        share a name, the second and later take a number after it, so each
        species keeps its own. The same file named twice lands once.
        """
        placed = self._placed()
        return placed[os.path.abspath(source)]

    def _placed(self) -> dict[str, str]:
        """Every species file's place under the tileset, by its absolute path."""
        sources = [source for entry in self.species
                   for source in (entry.mesh, entry.solidTexture,
                                  entry.foliageTexture, entry.impostor)]
        for entry in self._cover_species():
            sources.extend(part for part in (entry.card, entry.clump) if part)
        placed: dict[str, str] = {}
        taken: set[str] = set()
        for source in map(os.path.abspath, sources):
            if source in placed:
                continue
            stem, extension = os.path.splitext(os.path.basename(source))
            name, number = stem + extension, 1
            while name in taken:
                number += 1
                name = '%s-%d%s' % (stem, number, extension)
            taken.add(name)
            placed[source] = '%s/%s' % (SPECIES_DIRECTORY, name)
        return placed
