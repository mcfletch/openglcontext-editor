"""The ground a world is built on: a base, and the edits made to it.

A world's height was once a single pure function, which is enough for a
landscape nobody authors. A landscape a designer works on is two things: a
**base** -- the shipped procedural terrain, a preset, an imported elevation
dataset -- and an ordered stack of **edits** on top of it, each a bounded
change to a region. Raising a hill, carving a river bed and importing a real
valley are then the same kind of thing, and all three survive being written to
a project file and read back.

:meth:`HeightSource.height_fn` hands back one ordinary height function, so
everything that samples terrain -- the tile mesher, the road generator, the
scatter, a car asking where the ground is -- is pointed at this and needs to
know nothing about how it was composed.

**Every edit is vectorised and bounded.** It declares the rectangle it can
affect and is asked only about samples inside it, and it answers for all of
them at once. A bake samples the height function across whole tiles and the
editor across its whole plan view, so an edit that looped in Python -- or that
was consulted about ground a kilometre away -- would put the cost of authoring
into every frame and every tile of every world afterwards.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from OpenGLContext.loaders.tiles3d.procedural import terrain_height

__all__ = ['HeightBase', 'HeightEdit', 'HeightSource', 'ProceduralBase',
           'base_from_json', 'edit_from_json', 'register_base', 'register_edit']

#: What everything downstream of a height source is handed.
HeightFn = Callable[[Any, Any], np.ndarray]

#: How tall the shipped landscape's hills are by default, as a multiple of its
#: own relief. At 1 it rises five hundred metres over four kilometres, which no
#: road held to a drivable grade can follow.
DEFAULT_RELIEF = 0.5


class HeightBase(ABC):
    """The land before anybody edited it."""

    #: What this base is called in a project file.
    KIND: ClassVar[str] = ''

    @abstractmethod
    def sample(self, x: Any, z: Any) -> np.ndarray:
        """The height at each ``(x, z)``, in metres, answering all at once."""

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """This base as the project file holds it."""

    @classmethod
    @abstractmethod
    def from_json(cls, document: dict[str, Any]) -> HeightBase:
        """A base back from what :meth:`to_json` wrote."""


class HeightEdit(ABC):
    """One bounded change to the ground, on top of whatever is under it."""

    KIND: ClassVar[str] = ''

    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """The ground this edit can reach, as ``(min_x, min_z, max_x, max_z)``.

        Outside it the edit contributes nothing and is not asked, so an edit
        that covers a hundred metres of a four-kilometre world costs the rest
        of the world one rectangle comparison.
        """

    @abstractmethod
    def delta(self, x: Any, z: Any, height: Any) -> np.ndarray:
        """How much to add at each sample, given the ground already there.

        Only the samples inside :meth:`bounds` are passed, as flat arrays of
        equal length. ``height`` is what everything before this edit made of
        the ground, so an edit can lift the land it finds rather than replace
        it.
        """

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """This edit as the project file holds it, ``kind`` included."""

    @classmethod
    @abstractmethod
    def from_json(cls, document: dict[str, Any]) -> HeightEdit:
        """An edit back from what :meth:`to_json` wrote."""


@dataclass(frozen=True)
class ProceduralBase(HeightBase):
    """The landscape the toolkit ships, at a chosen relief.

    ``relief`` multiplies its height: halved, the same hills, canyon and lake
    basin make country a road can be built through, with a handful of crossings
    where it still cannot.
    """

    KIND: ClassVar[str] = 'procedural'
    relief: float = DEFAULT_RELIEF

    def sample(self, x: Any, z: Any) -> np.ndarray:
        ground = np.asarray(terrain_height(x, z), dtype='d')
        if self.relief == 1.0:
            return ground
        return ground * float(self.relief)

    def to_json(self) -> dict[str, Any]:
        return {'kind': self.KIND, 'relief': float(self.relief)}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> ProceduralBase:
        return cls(relief=float(document.get('relief', DEFAULT_RELIEF)))


#: The base and edit kinds a project file may name. A kind that is not here is
#: refused rather than dropped: a file half-read loses a designer's work
#: without saying so.
BASE_KINDS: dict[str, type[HeightBase]] = {}
EDIT_KINDS: dict[str, type[HeightEdit]] = {}


def register_base(kind: type[HeightBase]) -> type[HeightBase]:
    """Declare a base a project file may name. Usable as a decorator."""
    BASE_KINDS[kind.KIND] = kind
    return kind


def register_edit(kind: type[HeightEdit]) -> type[HeightEdit]:
    """Declare an edit a project file may name. Usable as a decorator."""
    EDIT_KINDS[kind.KIND] = kind
    return kind


register_base(ProceduralBase)


def base_from_json(document: dict[str, Any]) -> HeightBase:
    """The base a project file describes."""
    kind = str(document.get('kind', ProceduralBase.KIND))
    if kind not in BASE_KINDS:
        raise ValueError(
            "this landscape is built on %r, which this version does not know "
            "how to make" % kind)
    return BASE_KINDS[kind].from_json(document)


def edit_from_json(document: dict[str, Any]) -> HeightEdit:
    """One edit a project file describes."""
    kind = str(document.get('kind', ''))
    if kind not in EDIT_KINDS:
        raise ValueError(
            "this landscape carries a %r edit, which this version does not "
            "know how to make" % kind)
    return EDIT_KINDS[kind].from_json(document)


@dataclass
class HeightSource:
    """A base and the edits made to it, as one height function.

    The edits are applied in order, each on the ground the ones before it left,
    which is what makes "raise this hill, then run a river down it" mean what a
    designer expects.
    """

    base: HeightBase = field(default_factory=ProceduralBase)
    edits: list[HeightEdit] = field(default_factory=list)

    def height_fn(self) -> HeightFn:
        """The ground as this source finally has it.

        Built fresh rather than cached: the edit stack is what a designer is
        changing, and a function held across an edit would answer about the
        landscape as it used to be.
        """
        base = self.base
        edits = tuple(self.edits)

        def sample(x: Any, z: Any) -> np.ndarray:
            height = np.asarray(base.sample(x, z), dtype='d')
            if not edits:
                return height
            shape = height.shape
            flat_x = np.broadcast_to(np.asarray(x, dtype='d'), shape)
            flat_z = np.broadcast_to(np.asarray(z, dtype='d'), shape)
            ours = False
            for edit in edits:
                low_x, low_z, high_x, high_z = edit.bounds()
                inside = ((flat_x >= low_x) & (flat_x <= high_x)
                          & (flat_z >= low_z) & (flat_z <= high_z))
                if not inside.any():
                    continue
                if not ours:
                    # The base's answer may be something it keeps; the edits
                    # are written into a copy of it rather than through it.
                    height = height.copy()
                    ours = True
                height[inside] += np.asarray(
                    edit.delta(flat_x[inside], flat_z[inside], height[inside]),
                    dtype='d')
            return height

        return sample

    # -- the file ----------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {'base': self.base.to_json(),
                'edits': [edit.to_json() for edit in self.edits]}

    @classmethod
    def from_json(cls, document: dict[str, Any]) -> HeightSource:
        """A source back from a project file.

        A file with no ``base`` reads as the shipped landscape, so a track
        written before there was a source block opens as it always did.
        """
        return cls(base=base_from_json(document.get('base', {})),
                   edits=[edit_from_json(entry)
                          for entry in document.get('edits', ())])
