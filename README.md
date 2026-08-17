# OpenGLContext-editor

World-authoring toolkit for [OpenGLContext](https://github.com/mcfletch/openglcontext).

OpenGLContext renders and streams a world. This package makes one: it takes an
authored scene — terrain, roads, water, scattered vegetation — and bakes it into
a 3D Tiles octree of glTF content that the engine's existing streaming runtime
loads.

Three things live here, and they share one property: every *editor* wants them
identically, and no shipped *game* wants them in its dependency tree.

- **The tile baker.** Spatial partition of an authored world into an octree,
  a level of detail per node, and the tileset written out through the engine's
  glTF and 3D Tiles writers.
- **World generation.** Road and water geometry, DEM to splat control map,
  refinement driven by distance to a road.
- **The editor UI toolkit.** Tool modes, menus, gizmos and a top-down ortho map,
  built on `OpenGLContext.ui`.

A game imports `OpenGLContext`. An editor imports both.

## Install

The package is developed inside the
[OpenGL-dev workspace](https://github.com/mcfletch/OpenGL-dev), where a
`uv sync` at the workspace root installs it editable alongside the engine.
Standalone:

```bash
pip install -e ".[dev]"
pytest
```

## Layout

| Path | Holds |
|---|---|
| `src/OpenGLContext_editor/` | the package |
| `tests/` | the suite; `pytest` runs it |
| `specs/` | format and interoperability facts the code cites, and the [clean-room procedure](specs/CLEAN-ROOM.md) that governs how they are gathered |

## Status

Bootstrapped. The baker and the generation phases are designed in
[GLISTEEL-WORLD-AUTHORING.md](https://github.com/mcfletch/openglcontext/blob/main/plans/GLISTEEL-WORLD-AUTHORING.md),
which also records the division of labour between this package and the engine.

## Licence

BSD-3-Clause; see [license.txt](license.txt).
