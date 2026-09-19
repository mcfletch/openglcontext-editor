"""``MSFT_lod``: which nodes are coarser selves of which, and from when.

The extension is written in node *indices*, and those exist only once the
exporter has laid the document out. So authoring a chain and writing one are
two different jobs: in Blender an object says which chain it belongs to and how
fine it is, and at the end of an export that becomes "node 7 names 8, 9 and 10
as its coarser selves, and those three belong to no scene".

This module is the second job, and it holds no Blender. :func:`plan` turns a
set of :class:`Level` records into a :class:`Plan`, and :func:`apply` writes
that plan into a glTF document. What the spec fixes:

* the node carrying the extension is the **finest** level;
* ``ids`` lists the coarser ones in **decreasing** detail;
* ``MSFT_screencoverage`` goes in the node's ``extras`` with one value per
  level, the finest first, and says what share of the window's height a level
  takes over at.

Reference:
    https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/MSFT_lod
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'COVERAGE',
    'COVERAGE_PROPERTY',
    'EXTENSION',
    'FIRST_COVERAGE',
    'GROUP_PROPERTY',
    'LEVEL_PROPERTY',
    'Level',
    'Plan',
    'apply',
    'coverage_series',
    'level_of',
    'named_level',
    'plan',
    'write_into',
]

#: What a node's ``extensions`` calls the list of coarser levels.
EXTENSION = 'MSFT_lod'
#: What its ``extras`` calls the thresholds they switch at.
COVERAGE = 'MSFT_screencoverage'

#: Where a guessed series starts, and what each further level takes over at.
#: Halving matches what OpenGLContext's reader assumes of a file that states
#: nothing, so a chain exported without measured thresholds behaves the same
#: whether the numbers travel in the file or not.
FIRST_COVERAGE = 0.5

#: The custom properties an object carries to say where it sits in a chain.
GROUP_PROPERTY = 'lod_group'
LEVEL_PROPERTY = 'lod_level'
COVERAGE_PROPERTY = 'lod_coverage'

#: ``Bust_LOD2``, ``Bust.LOD2``, ``Bust-LOD2`` -- the suffix the ecosystem
#: writes chains with. The stem must not be empty and the number must end the
#: name, so Blender's own ``Bust.001`` duplicate suffix is not mistaken for one.
_SUFFIX = re.compile(r'^(?P<stem>.+?)[._-]lod(?P<level>\d+)$', re.IGNORECASE)


def named_level(name: str) -> tuple[str, int] | None:
    """``('Bust', 2)`` for ``'Bust_LOD2'``; None for a name that says nothing."""
    found = _SUFFIX.match(name)
    if found is None:
        return None
    return found.group('stem'), int(found.group('level'))


def coverage_series(count: int, first: float = FIRST_COVERAGE) -> list[float]:
    """Thresholds for ``count`` levels when nobody measured any.

    Each level takes over at half the share of the one before it, and the
    coarsest at nothing at all: a threshold a *reader* guessed must not be the
    reason a model stops being drawn.
    """
    if count <= 0:
        return []
    return [first / (2 ** index) for index in range(count - 1)] + [0.0]


@dataclass(frozen=True)
class Level:
    """One level of one chain, as the scene declares it.

    ``level`` orders the chain and does not index it: an author who deleted the
    middle of a chain has a shorter chain, not a hole in one.
    """

    node: int
    group: str
    level: int
    #: What share of the window's height this level takes over at, where the
    #: author measured it. None leaves the whole chain to :func:`coverage_series`.
    coverage: float | None = None


@dataclass(frozen=True)
class Plan:
    """What an exported document has to be told, in its own node indices."""

    #: Finest node -> its coarser selves, in decreasing detail.
    ids: dict[int, list[int]] = field(default_factory=dict)
    #: Finest node -> one threshold per level, itself first.
    coverage: dict[int, list[float]] = field(default_factory=dict)
    #: Every node that is an alternative, and so belongs to no scene.
    detach: frozenset[int] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.ids)


def plan(levels: Iterable[Level]) -> Plan:
    """Group ``levels`` into chains and say what each becomes.

    A group with one level in it is an ordinary object and gets no extension.
    """
    chains: dict[str, list[Level]] = {}
    for one in levels:
        chains.setdefault(one.group, []).append(one)

    ids: dict[int, list[int]] = {}
    coverage: dict[int, list[float]] = {}
    detach: set[int] = set()
    for group, members in chains.items():
        members = sorted(members, key=lambda one: one.level)
        _refuse_repeats(group, members)
        if len(members) < 2:
            continue
        finest, rest = members[0], members[1:]
        ids[finest.node] = [one.node for one in rest]
        coverage[finest.node] = _thresholds(group, members)
        detach.update(one.node for one in rest)
    return Plan(ids=ids, coverage=coverage, detach=frozenset(detach))


def _refuse_repeats(group: str, members: Sequence[Level]) -> None:
    seen = [one.level for one in members]
    if len(set(seen)) != len(seen):
        raise ValueError(
            'chain %r has more than one object at the same level: %r'
            % (group, sorted(seen))
        )


def _thresholds(group: str, members: Sequence[Level]) -> list[float]:
    """One value per level: the author's where they gave them, else a series.

    Some of them is not an answer. A chain where three levels name a threshold
    and one does not is a chain whose author meant something the file cannot
    say, and filling the gap by halving would bury that in a number that looks
    deliberate.
    """
    stated = [one.coverage for one in members]
    if all(value is None for value in stated):
        return coverage_series(len(members))
    if any(value is None for value in stated):
        raise ValueError(
            'chain %r states a screen coverage for some of its levels and not '
            'others; state one for every level or for none' % (group,)
        )
    values = [float(value) for value in stated]        # type: ignore[arg-type]
    for finer, coarser in zip(values, values[1:], strict=False):
        if coarser > finer:
            raise ValueError(
                'chain %r takes over at a larger share of the screen the '
                'coarser it gets: %r' % (group, values)
            )
    return values


def level_of(name: str, properties: Any, node: int) -> Level | None:
    """What ``node`` is a level of, from an object's properties and its name.

    ``properties`` is anything answering ``get``, which a Blender object does.
    Its custom properties are the author's own word and are taken over the
    name; a name in the ``Bust_LOD2`` form answers where they are absent, so a
    chain brought in from elsewhere needs nothing done to it. An object that
    says neither is not part of a chain.
    """
    group = properties.get(GROUP_PROPERTY)
    level = properties.get(LEVEL_PROPERTY)
    if group is None:
        named = named_level(name)
        if named is None:
            return None
        group, named_index = named
        if level is None:
            level = named_index
    return Level(node=node, group=str(group),
                 level=_whole('level', level, group),
                 coverage=_fraction(properties.get(COVERAGE_PROPERTY), group))


def _whole(what: str, value: Any, group: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError('%r on chain %r is not a %s number: %r'
                         % (LEVEL_PROPERTY, group, what, value)) from None


def _fraction(value: Any, group: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError('%r on chain %r is not a share of the screen: %r'
                         % (COVERAGE_PROPERTY, group, value)) from None


def apply(document: dict, made: Plan) -> dict:
    """Write ``made`` into a glTF ``document``, in place."""
    _write(document, made)
    return document


def write_into(gltf: Any, made: Plan,
               keep_declared: list[str] | None = None) -> Any:
    """Write ``made`` into the glTF exporter's own objects, in place.

    Blender's exporter hands an extension the document as objects rather than
    as JSON, and prunes any declaration it was not told to keep --
    ``keep_declared`` is that list.
    """
    _write(gltf, made, keep_declared=keep_declared)
    return gltf


def _write(document: Any, made: Plan,
           keep_declared: list[str] | None = None) -> None:
    """The rules, over either a glTF document or the exporter's objects.

    The alternatives are taken out of every scene and every node's children,
    because a level that is still reachable is a level that is still drawn --
    which would put the whole chain on screen at once, in one place.
    """
    if not made:
        return
    nodes = _get(document, 'nodes') or []
    for finest, alternatives in made.ids.items():
        node = nodes[finest]
        _into(node, 'extensions')[EXTENSION] = {'ids': list(alternatives)}
        _into(node, 'extras')[COVERAGE] = list(made.coverage[finest])
    _detach(document, nodes, made.detach)
    for declaration in (_declared(document), keep_declared):
        if declaration is not None and EXTENSION not in declaration:
            declaration.append(EXTENSION)


def _detach(document: Any, nodes: Sequence[Any], gone: frozenset[int]) -> None:
    for scene in _get(document, 'scenes') or []:
        kept = [index for index in (_get(scene, 'nodes') or [])
                if index not in gone]
        _set(scene, 'nodes', kept)
    for node in nodes:
        children = _get(node, 'children')
        if children:
            kept = [index for index in children if index not in gone]
            # glTF gives ``children`` a minimum of one entry, so a node whose
            # every child was an alternative has no children rather than none.
            _set(node, 'children', kept or None)


def _declared(document: Any) -> list[str] | None:
    """``extensionsUsed``, under whichever of its two spellings this has."""
    if isinstance(document, dict):
        declared: list[str] = document.setdefault('extensionsUsed', [])
        return declared
    used: list[str] | None = getattr(document, 'extensions_used', None)
    if used is None and hasattr(document, 'extensions_used'):
        document.extensions_used = used = []
    return used


def _get(holder: Any, key: str) -> Any:
    if isinstance(holder, Mapping):
        return holder.get(key)
    return getattr(holder, key, None)


def _set(holder: Any, key: str, value: Any) -> None:
    if isinstance(holder, dict):
        if value is None:
            holder.pop(key, None)
        else:
            holder[key] = value
    else:
        setattr(holder, key, value)


def _into(holder: Any, key: str) -> dict:
    """The dict at ``key``, made if it is absent or None."""
    found: dict = _get(holder, key)
    if found is None:
        found = {}
        _set(holder, key, found)
    return found
