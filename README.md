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

## Bake one now

```bash
pip install OpenGLContext-editor
oglc-bake --output /tmp/world      # hills, a canyon, a lake, a conifer forest
oglc-view /tmp/world/tileset.json  # walk around in it
```

`oglc-bake --view` runs both steps. `--extent`, `--depth` and `--resolution`
set how big the world is, how deep the tile tree refines and how many ground
samples each tile spends; `oglc-bake --help` lists the rest.

## Describe a world of your own

A world is a list of *layers*, and a layer answers one question: what is in this
region, at this level of detail?

```python
from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.driver import bake_world
from OpenGLContext_editor.bake.layers import HeightfieldLayer, InstanceLayer
from OpenGLContext_editor.world.scatter import scatter_on_heightfield, yaw_quaternions

ground = BoundingBox((-2048, 0, -2048), (2048, 0, 2048))
terrain = HeightfieldLayer(height_fn=my_heights, extent=ground, resolution=33,
                           color_fn=my_colours, water_level=0.0)

trees = scatter_on_heightfield(my_heights, ground, density=0.004, seed=11,
                               slope_limit=38.0, height_range=(2.0, 130.0))
forest = InstanceLayer(positions=trees.positions,
                       rotations=yaw_quaternions(trees.yaws),
                       scales=trees.scales,
                       lods=[(0.0, conifer), (8.0, impostor)])

print(bake_world([terrain, forest], '/tmp/world', depth=4).summary())
```

The full account of what the baker does and why -- the partition, the level-of-
detail policy, the geometric-error ladder, and the limits -- is in the engine's
[Baking a world](https://github.com/mcfletch/openglcontext/blob/main/docs/baking.html)
page.

## Install for development

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
| `src/OpenGLContext_editor/bake/` | the tile baker: bounds, the octree, layers, the tileset writer, the bake driver |
| `src/OpenGLContext_editor/world/` | world generation: scatter, and the example world |
| `src/OpenGLContext_editor/bin/` | `oglc-bake` |
| `tests/` | the suite; `pytest` runs it |
| `specs/` | format and interoperability facts the code cites, and the [clean-room procedure](specs/CLEAN-ROOM.md) that governs how they are gathered |

## Status

The baker works: a world of terrain and instanced vegetation bakes to a
3D Tiles octree the engine streams. Roads, water, and the editor UI toolkit are
designed in
[GLISTEEL-WORLD-AUTHORING.md](https://github.com/mcfletch/openglcontext/blob/main/plans/GLISTEEL-WORLD-AUTHORING.md),
which also records the division of labour between this package and the engine.

## Licence

BSD-3-Clause; see [license.txt](license.txt).
