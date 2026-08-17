"""Generating what a world is made of, before any of it is baked.

Where :mod:`OpenGLContext_editor.bake` turns content into tiles, this turns a
designer's intent into content: where the trees stand, where the road runs,
where the water sits. Everything here is pure geometry and arrays -- no tiles,
no GL -- so a world can be generated, inspected and asserted on without writing
a file.

``scatter``   placing instances over a height field, filtered by slope,
              elevation and any mask a world cares to supply
"""
