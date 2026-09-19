"""``MSFT_lod`` on the way out of Blender's own glTF exporter.

Blender's exporter asks every enabled add-on for a ``glTF2ExportUserExtension``
and calls it at points through the export. Two of those points are enough:

``gather_node_hook``
    once per exported object, where the Blender object -- and so the custom
    properties saying which chain it is a level of -- is still in hand;

``gather_gltf_extensions_hook``
    once, with the whole document laid out, which is the first moment a node
    *has* an index and so the first moment ``MSFT_lod`` can be written at all.

What is written between them is :mod:`msftlod`'s, which holds no Blender and is
tested without it. This module is the wiring, and its own job is to be
suspicious: the exporter runs a hook inside a ``try`` and logs what it raises,
so an extension that fails quietly produces a file that is wrong in a way
nobody is told about. Anything refused here is said out loud.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from . import msftlod

__all__ = ['MSFTLODExtension']


class MSFTLODExtension:
    """Turns the scene's chains into ``MSFT_lod`` as the document is finished."""

    def __init__(self) -> None:
        #: id(exported node) -> the level it is, its node index not yet known.
        self._levels: dict[int, msftlod.Level] = {}
        #: What could not be read, to be reported once rather than per object.
        self._refused: list[str] = []

    def gather_node_hook(self, gltf2_node: Any, blender_object: Any,
                         export_settings: dict) -> None:
        """Note whether this object is a level of something."""
        if blender_object is None or not hasattr(blender_object, 'get'):
            return
        try:
            level = msftlod.level_of(getattr(blender_object, 'name', ''),
                                     blender_object, node=0)
        except ValueError as error:
            self._refused.append(str(error))
            return
        if level is not None:
            self._levels[id(gltf2_node)] = level

    def gather_gltf_extensions_hook(self, gltf: Any,
                                    export_settings: dict) -> None:
        """Write every chain, now that the nodes have indices."""
        log = export_settings.get('log')
        for complaint in self._refused:
            _say(log, 'error', 'MSFT_lod: %s' % (complaint,))
        self._refused.clear()
        if not self._levels:
            return

        where = {id(node): index for index, node in enumerate(gltf.nodes or [])}
        found = [dataclasses.replace(level, node=where[key])
                 for key, level in self._levels.items() if key in where]
        missing = len(self._levels) - len(found)
        if missing:
            _say(log, 'warning',
                 'MSFT_lod: %d marked object(s) did not reach the document and '
                 'are not in any chain' % (missing,))
        self._levels.clear()
        if not found:
            return

        try:
            plan = msftlod.plan(found)
        except ValueError as error:
            _say(log, 'error', 'MSFT_lod: %s -- no levels were written'
                 % (error,))
            return
        msftlod.write_into(
            gltf, plan,
            keep_declared=export_settings.get(
                'gltf_need_to_keep_extension_declaration'),
        )
        _say(log, 'info', 'MSFT_lod: %d chain(s), %d alternative level(s)'
             % (len(plan.ids), len(plan.detach)))


def _say(log: Any, level: str, message: str) -> None:
    """Through the exporter's own log where there is one, else stdout.

    A headless build has somewhere for this to go either way, and a message
    nobody can see is the failure this module exists to avoid.
    """
    reporter = getattr(log, level, None) if log is not None else None
    if reporter is None:
        print(message)
    else:
        reporter(message)
