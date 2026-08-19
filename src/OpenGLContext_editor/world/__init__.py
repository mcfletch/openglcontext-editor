"""Generating what a world is made of, before any of it is baked.

Where :mod:`OpenGLContext_editor.bake` turns content into tiles, this turns a
designer's intent into content: where the trees stand, where the road runs,
where the water sits. Everything here is pure geometry and arrays -- no tiles,
no GL -- so a world can be generated, inspected and asserted on without writing
a file.

``scatter``   placing instances over a height field, filtered by slope,
              elevation and any mask a world cares to supply
"""

# The height bases and edits a project file may name have to be declared before
# anything reads one, and a file names a kind rather than a module. Importing
# them here is what makes ``base_from_json`` / ``edit_from_json`` know the kinds
# this package ships, whichever of its modules the caller reached for first.
from OpenGLContext_editor.world import hydrology as _hydrology  # noqa: E402,F401
from OpenGLContext_editor.world import presets as _presets  # noqa: E402,F401
from OpenGLContext_editor.world import sculpt as _sculpt  # noqa: E402,F401
