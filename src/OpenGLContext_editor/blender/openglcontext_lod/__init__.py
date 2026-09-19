"""Levels of detail in Blender, exported as ``MSFT_lod``.

A Blender add-on. Install it, and Blender gains:

* **Object > Levels of Detail > Make LOD chain** -- a chain from the selected
  mesh, cut by Blender's own Decimate modifier, each level named and marked;
* **``MSFT_lod`` on glTF export** -- the ordinary *File > Export > glTF 2.0*
  writes the chain as the vendor extension, so a viewer that knows it switches
  levels and one that does not draws the finest;
* **the bust gallery** -- the demo world, as a worked example of both.

Only Blender's own Python is needed; the modules that carry no Blender import
cleanly outside it, which is where they are tested.
"""

from __future__ import annotations

bl_info = {
    'name': 'OpenGLContext levels of detail',
    'author': 'Mike C. Fletcher',
    'version': (1, 0, 0),
    'blender': (4, 2, 0),
    'location': 'Object > Levels of Detail, and glTF 2.0 export',
    'description': 'Author LOD chains and export them as MSFT_lod',
    'doc_url': 'https://github.com/mcfletch/openglcontext-editor',
    'category': 'Import-Export',
}

try:
    import bpy as _bpy
except ImportError:                                  # outside Blender
    _bpy = None                                      # type: ignore[assignment]

#: What the glTF exporter is handed, once Blender is registering us. Kept at
#: module scope because the exporter asks the add-on module for it by name.
glTF2ExportUserExtension = None


def register() -> None:
    """Add the operators, the panel and the glTF export extension."""
    if _bpy is None:
        raise RuntimeError('this add-on runs inside Blender')
    from . import exporter, ops

    global glTF2ExportUserExtension
    glTF2ExportUserExtension = exporter.MSFTLODExtension
    ops.register()


def unregister() -> None:
    if _bpy is None:
        return
    from . import ops

    global glTF2ExportUserExtension
    glTF2ExportUserExtension = None
    ops.unregister()
