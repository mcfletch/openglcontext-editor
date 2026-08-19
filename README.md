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

trees = scatter_on_heightfield(my_heights, ground, spacing=2.6, seed=11,
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

## Where the ground comes from

A world's height is a **base** and an ordered stack of **edits** on it:

```python
from OpenGLContext_editor.world.height import HeightSource, ProceduralBase

source = HeightSource(base=ProceduralBase(relief=0.5))
ground = source.height_fn()          # an ordinary height function
```

`height_fn()` hands back one callable over `(x, z)`, so everything that samples
terrain — the tile mesher, the road generator, the scatter, a car asking where
the ground is — is pointed at it and needs to know nothing about how it was
composed. `ProceduralWorld(source=...)` takes one; a world given none is the
shipped landscape at its own `relief`.

An edit is a `HeightEdit`: a rectangle it can reach (`bounds()`), and how much
to add at each sample inside it (`delta(x, z, height)`), given the ground
everything before it left. Applying them in order is what makes "raise this
hill, then run a river down it" mean what a designer expects.

**Every edit is vectorised and bounded**, and both matter: a bake samples the
height function across whole tiles and an editor across its whole plan view, so
an edit that looped in Python — or that was consulted about ground a kilometre
away — would put the cost of authoring into every frame and every tile of every
world afterwards. Outside its rectangle an edit costs one comparison.

A source is JSON, so a designer's landscape survives being saved:

```json
{"base": {"kind": "procedural", "relief": 0.5}, "edits": []}
```

A `kind` this version does not know is **refused rather than dropped**: reading
half of a file loses work without saying so. Declare a new kind with
`register_base` / `register_edit`.

## Start from a landscape

The presets are tuned terrain profiles under a name and a sentence, so a game
editor built on this toolkit gets the same starting points:

```python
from OpenGLContext_editor.world.presets import PRESETS, PresetBase

source = HeightSource(base=PresetBase(name='canyon'))
print(PRESETS['canyon'].description)
```

| Preset | What you get |
|---|---|
| `shipped` | hill country with a range, a river canyon and a lake basin |
| `mountains` | ranges over most of the map, rising most of a kilometre |
| `lakes` | low country dished into broad basins that flood |
| `hills` | nothing steeper than a road can climb, anywhere |
| `canyon` | a gorge three hundred metres deep across a high plain |

`relief` multiplies a preset's height — which is how a landscape too tall for a
road becomes one a road can be built through without changing what it looks
like — and `seed` gives another landscape of the same description.

## Or from real ground

`DEMBase` puts an elevation file under a world at a point on the Earth:

```python
from OpenGLContext_editor.world.dem import DEMBase

source = HeightSource(base=DEMBase(path='N47E008.hgt', centre=(47.5, 8.5),
                                   datum=0.0))
```

`centre` is the `(latitude, longitude)` the world's origin stands on and is
recorded in the project, so a reopened track resolves to the same ground.
`datum` is the height that origin is given: real ground is hundreds of metres
above the sea, and a world whose waterline is at zero would have all of it
underwater. `relief` scales what is left, for a valley whose real sides no road
can climb.

**Format and limits.** SRTM `.hgt` — raw big-endian 16-bit samples, one degree
square, named for its south-west corner. Voids are filled from the ground around
them. The mapping from degrees to metres is a local tangent plane about the
centre, which holds over the few tens of kilometres a track covers and is not a
projection to use across a continent. **Nothing is fetched**: the file is one the
designer supplies. The facts behind all of it, with their sources, are in
[specs/ELEVATION-DATA.md](specs/ELEVATION-DATA.md).

## Shape it by hand

A `SculptStroke` is one gesture of a brush, recorded as data on the edit stack:

```python
from OpenGLContext_editor.world.sculpt import SculptStroke

source.edits.append(SculptStroke(centre=(120.0, -40.0), radius=150.0,
                                 amount=25.0, detail=0.35))
```

`amount` is metres at the centre, positive up; `falloff` is how sharply it dies
away towards `radius`, reaching zero *at* the radius so a stroke never steps;
and `detail` is how much of the lift is fractal variation rather than a smooth
dome. The detail is a **share of the lift**, so a gentle stroke gets gentle
detail and a raised hill sits in the same visual family as the land around it.

## Put water on it

From a **spring**, follow the ground downhill until the water reaches the
waterline, runs off the edge of the world, or arrives somewhere it cannot get
out of — which is where a lake is, and is an answer rather than a failure:

```python
from OpenGLContext_editor.world.hydrology import Spring, channels_for, flow_from

paths = flow_from(ground, [Spring(at=(-800.0, 700.0))], extent=2048.0,
                  water_level=0.0)
source.edits.extend(channels_for(paths))     # the beds the rivers cut
```

Rivers that meet **merge**: a path arriving on one already there stops and adds
its water to it, so the river below a junction cuts a wider, deeper bed than
either branch above it. Water also **fills a hollow and spills**: ground is not
a smooth ramp, and a river that stopped in the first dimple of a noisy hillside
would never reach anything.

The `Channel` is an ordinary height edit, so the river becomes part of the
ground: a road crosses it as water rather than as a stripe, and sculpting the
land upstream reroutes it the next time the flow is worked out. The path is
worth keeping and the bed is not — a bed written to a file is the river as the
land *used to be*.

**Limits.** The water surface is the world's own water layer, so a river is a
carved bed with water in it only where it runs below the waterline; a stream
running down a mountainside is a valley, not a ribbon of water. Flow is
steepest-descent from a point, not a catchment model: it says where water from
*here* goes, not how much of it there is.

## Read the land off it

Iso-height lines over a height field, for a plan view to draw and a road to be
held to a grade against:

```python
from OpenGLContext_editor.world.contours import contours_of

for contour in contours_of(ground, extent=2048.0, interval=25.0, resolution=257):
    print(contour.elevation, len(contour.lines))   # (N,2) xz polylines
```

`interval` is the spacing in metres and `resolution` how finely the field is
sampled to find the lines — the detail of the drawn line, and what it costs: the
height function is asked once, for the whole grid. A line that comes back to
where it started is a closed loop (a hilltop or a basin); one that does not runs
off the edge of the ground. `contours(heights, min_x, max_x, min_z, max_z, ...)`
takes an already-sampled grid instead.

The extraction is marching squares, so it is exact for a field that is linear
inside a cell and converges on the truth as the sampling gets finer. It is for
*drawing*: nothing here is a substitute for asking the height function itself
where a particular elevation is.

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

### What the road puts beside itself

A generated road knows what it is about to do, so the roadside is derived rather
than authored. `world.signs.warn_of` reads curvature, grade and structures off
the alignment and returns the warnings it wants, a stopping distance before each
hazard; `bake.signs.SignLayer` writes them.

A circuit also needs its lap to be visible. `world.gantry.start_finish` takes the
crown where the centreline begins, spans the carriageway and its shoulders, and
measures the ground under each leg; `bake.gantry.GantryLayer` writes the frame
and the chequered line painted under it as one mesh, and the two legs into the
world's props so a car can hit them.

```python
from OpenGLContext_editor.bake.gantry import GantryLayer
from OpenGLContext_editor.world.gantry import start_finish

GantryLayer(placement=start_finish(path, ground=my_heights))
```

## Corners a designer drew

A drawn plan's vertices are corners: the road turns through the whole of one at
a single point, which no car can take. There are two ways to answer that, and
which one is right depends on where the corner came from:

| | |
|---|---|
| `hold_radius` | relaxes the line towards its chords until nothing is too tight — right for a route being **found**, where the corner is an artefact of the search |
| `hold_corners` | rounds each corner **in place**, with a circular fillet tangent to both legs — right for a route that was **drawn**, where the corner is the point |

A **switchback** is the case that decides it. A hairpin drawn up a mountainside
is the only way to gain height where the slope is steeper than a road can
climb; relaxing it puts the road somewhere else, while rounding it leaves the
legs where the designer drew them and the hairpin still a hairpin — of the
tightest radius a car can take, or the tightest the legs have room for.

`ProceduralWorld` applies `hold_corners` to any route it is given and
`ease_route` to the circuit it invents for itself, which is the same
distinction. A vertex turning less than fifteen degrees is a sample of a curve
rather than a corner, and is left alone.

Where a lap begins travels with the road: `ProceduralWorld(start_at=(x, z))`
puts the gantry at the station nearest that point, and the tileset's `extras`
carry it as `roads[0]['start']` — a distance along the centreline, which is what
the grid, the timing and the autopilot count from.

## Baking is meant to be iterated

Baking the shipped four-kilometre world -- half a million trees, its roads and
their structures, its signs, its start line and its boulders -- takes about
**27 seconds**. That
is the number that matters: a world nobody can re-bake is a world nobody
revises, and everything in this package exists so that a designer can change a
route and drive it.

Nearly all of it used to be the tree scatter, and nearly all of *that* was
questions asked about ground the answer did not depend on. Three things fixed
it, and each is a rule worth following in a layer of your own:

- **Ask for a spacing, not a density,** for anything that will be thinned to a
  minimum separation afterwards. Random candidates have to be several times too
  dense before the thinning saturates, and each one is paid for.
- **Run the filters cheapest-first, each on what the last one left.** An
  elevation band is free once the height is known; a slope is four more height
  lookups apiece.
- **Do not ask the road about ground it never reaches.** `RoadPath`'s index
  drops whole cells of a query in one pass rather than iterating over the
  hundreds of thousands a fine grid makes.

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
| `src/OpenGLContext_editor/world/` | world generation: the height source, presets, DEM import, sculpting, hydrology, contours, scatter, roads, and the example world |
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

## What a bake writes beside the tiles

Every bake writes `world.json` next to the tileset: what the world is called,
its seed and extent, how long its road is, and how many metres of that road are
carried on bridges, bores and causeways. That is what anything offering a
*choice* of worlds reads, since reading a tileset to find out means loading the
world being chosen between. `--name` sets the name; without it a world is named
after the directory it was baked into, so `--output ashdown-forest` gives
*Ashdown Forest*.

The format is the engine's — `OpenGLContext.loaders.tiles3d.manifest` — so
anything that loads a baked world can read one without depending on this
authoring package. See `openglcontext/docs/baking.html`.
