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
- **World generation.** Road geometry and the alignment behind it — a drawn line
  slid onto ground a road can follow, settled for grade and cornering speed, and
  built as carriageway, causeway, viaduct or bore according to what the land
  under it will take. Water, DEM to splat control map, and refinement driven by
  distance to a road.
- **The editor UI toolkit.** Tool modes, menus, gizmos and a top-down ortho map,
  built on `OpenGLContext.ui`.

A game imports `OpenGLContext`. An editor imports both.

## Bake one now

```bash
pip install OpenGLContext-editor
oglc-bake --output /tmp/world      # hill country, a canyon, a lake, a forest
oglc-view /tmp/world/tileset.json  # walk around in it
```

`oglc-bake --view` runs both steps. `--extent`, `--depth` and `--resolution`
set how big the world is, how deep the tile tree refines and how many ground
samples each tile spends; `oglc-bake --help` lists the rest.

**How the ground and the trees are carried** is the choice that decides what a
world costs to draw. `--ground field` writes the landscape once beside the
tileset as a height image and a splat control map, which a viewer draws as one
mesh with detail materials blended per pixel; `--ground tiles` meshes it into
the tile tree instead, which is what a world too large to hold at once needs.
`--forest field` writes the trees as one table and lets the runtime choose what
to draw from how far each is from the camera; `--forest tiles` puts them in the
tiles. Both default to `field`, which is what the shipped world wants.

A `field` forest carries three more things with it. The trees' own **ground
cover** travels as a recipe rather than a scatter — a clump, a card, how dense,
and which of the splat map's layers it grows on — because there is far too much
ground to write a blade of grass for every square metre of it. The **canopy's
shade** is worked out from the trunks and read by the ground, the trees, the
grass and the road alike, so a clearing and a forest floor differ for all of
them at once. And the **road's own shade** is written into its surface, because
the trees do not move and neither does the sun.

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

## Put a road through it

A `RoadLayer` carries a route across the world: it cuts the ground to meet the
shoulder, splits the road so each tile owns the length inside it, and writes the
centreline into the tileset so a game can find the track in what it streams.

```python
from OpenGLContext_editor.world.road import (
    RoadPath, RoadLayer, conform_terrain_at, follow_terrain,
)

course = follow_terrain(my_route, my_heights, spacing=5.0,
                        maximum_grade=0.075, design_speed=47.0,
                        minimum_height=2.5, closed=True)
path = RoadPath(course)
ground = conform_terrain_at(my_heights, path)   # height fn -> height fn, per tile

terrain = HeightfieldLayer(height_fn_at=ground, extent=extent, resolution=33)
print(bake_world([terrain, RoadLayer(path=path)], '/tmp/world', depth=4).summary())
```

`follow_terrain` drapes a 2D route over the land and then makes it drivable:

- **smoothed**, so the road carries the shape of the landscape rather than its
  every hummock;
- **held to `maximum_grade`**, wrapping round the join for a closed circuit;
- **rounded off for `design_speed`** (metres per second), so no change of grade
  is sharp enough to take the car's wheels off the road at that speed. A grade
  limit alone permits a road that climbs at its limit and descends at its limit
  a few metres later, and a car meeting that at speed is launched, because there
  is nothing under it. This is the vertical curve a real road has;
- **lifted onto a causeway** where it would otherwise run below
  `minimum_height`, with its approaches raised to meet it.

`conform_terrain_at` returns the ground *with the road built into it*, at
whatever sample spacing the tile being baked uses -- a cut narrower than that
spacing falls between two vertices and never appears in the mesh.

Where the alignment is a few metres over low ground the road is instead carried
on a **causeway**: fill retained at the width of the road, walled at each edge,
with the ground either side left where it was found. The wall is low enough for
a seated driver to see over &mdash; a causeway is built to cross something worth
seeing.

An alignment that is not on the ground and not on a causeway is on an
**earthwork**: fill runs down
from the shoulder to where it meets the land, a cutting runs up to it, and how
far out that is depends on how far the road is from the ground and on nothing
else. A road already on the land disturbs almost nothing; one carried forty
metres over a valley builds an embankment as wide as it needs. Under the
carriageway the ground sits a hand's breadth below the surface, because a road
is built on a formation and surfaced on top of it.

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
| `src/OpenGLContext_editor/world/` | world generation: scatter, roads, and the example world |
| `src/OpenGLContext_editor/bin/` | `oglc-bake` |
| `tests/` | the suite; `pytest` runs it |
| `specs/` | format and interoperability facts the code cites, and the [clean-room procedure](specs/CLEAN-ROOM.md) that governs how they are gathered |

## Status

The baker works: a world of terrain, forest, ground cover and roads bakes to a
3D Tiles octree the engine streams, and [glisteel](https://github.com/mcfletch/glisteel)
drives a lap of it. A road follows the ground, rides a walled causeway over low
ground, spans a valley on a viaduct or runs through a bore, and carries the
shade of the wood it runs through. Water and the editor UI toolkit are designed
in
[GLISTEEL-WORLD-AUTHORING.md](https://github.com/mcfletch/openglcontext/blob/main/plans/GLISTEEL-WORLD-AUTHORING.md),
which also records the division of labour between this package and the engine.

## Licence

BSD-3-Clause; see [license.txt](license.txt).
