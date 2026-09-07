"""World-authoring toolkit for OpenGLContext.

OpenGLContext renders and streams a world; this package *makes* one. It holds
the half of the authoring pipeline no shipped game needs in its dependency
tree:

* the **tile baker** — spatial partition of an authored world into a 3D Tiles
  octree, with per-node level of detail, written through the engine's glTF and
  tileset writers;
* **world generation** — road and water geometry, DEM to splat control map,
  road-aware refinement;
* the **editor UI toolkit** — tool modes, menus, gizmos and the top-down map
  view, built on :mod:`OpenGLContext.ui`.

A game imports OpenGLContext alone. An editor imports both. The design, its
phases and the division of labour between the two packages are in
`GLISTEEL-WORLD-AUTHORING.md
<https://github.com/mcfletch/openglcontext/blob/main/plans/GLISTEEL-WORLD-AUTHORING.md>`_.
"""

__version__ = "1.0.0a1"
__author__ = "Michael Colin Fletcher"
__license__ = "BSD-Style, see license.txt for details"
