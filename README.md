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

The shipped world is a racing circuit in hill country, so the command that
bakes it belongs to the game's tools:

```bash
pip install glisteel-editor
glisteel-bake --output /tmp/world   # hill country, a canyon, a lake, a forest
oglc-view /tmp/world/tileset.json   # walk around in it
```

`glisteel-bake --view` runs both steps. `--extent`, `--depth` and
`--resolution` set how big the world is, how deep the tile tree refines and how
many ground samples each tile spends; `glisteel-bake --help` lists the rest.

The baker itself is here, and takes any layers at all:
`OpenGLContext_editor.bake.driver.bake_world`. `oglc-bake` remains for one
release cycle, saying where the command went.

**How the ground and the trees are carried** is the choice that decides what a
world costs to draw. `--ground field` writes the landscape once beside the
tileset as a height image and a splat control map, which a viewer draws as one
mesh with detail materials blended per pixel; `--ground tiles` meshes it into
the tile tree instead, so the ground gets finer as the tree refines — which is
what a world too large to hold at once needs, and where geometry a bake made
(rather than a surface dressed up at draw time) belongs.

The landscape is written beside the tileset **either way**, and
`extras.terrain.drawn` says which of the two draws it. A tiled ground is blended
and lit from that landscape — its primitives are named `ground`, and a viewer
that knows the name draws them with the world's own ground materials — and it is
the surface a game collides against, clamps a camera to and seats a plant on,
which is the surface that must not change resolution under a wheel as a tile
refines.
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

## Levels of detail, made once and shipped

A model arrives at whatever density its author left it, and a game cannot afford
that density at every distance. `meshlod` makes the coarser versions:

```python
from OpenGLContext_editor.meshlod import build_chain, measure_chain, write_chain

chain = build_chain(attributes, indices, levels=5)
write_chain('bust.glb', chain)          # bust.glb + bust.lod0.bin, ...
```

The whole reduction is decimated **once** — `opengl_decimate` records the
ordered contractions and each level is a prefix of that record — so five levels
cost one reduction and a sixth afterwards costs nothing. Normals are carried
through rather than recomputed, so a flattened triangle still shades the way the
surface it replaced did.

`write_chain` writes the levels as a glTF the whole ecosystem reads:
`MSFT_lod` names them, the coarsest rides inside the glb so the file always
draws something, and each finer level is a sidecar the operating system never
opens until it is wanted. The engine reads that back on its own — see
[Levels of detail](https://github.com/mcfletch/openglcontext/blob/main/docs/gltf.html)
— so a baked chain needs nothing of this package at play time. Nothing here is:
decimating two hundred assets when a player opens a door is not a thing that can
be done, which is why it is a bake.

**A level is judged by rendering it.** A geometric error is a length, and a
length says nothing on its own — a millimetre is invisible on a building and
ruinous on a face. `measure_chain` draws each level against the original over a
sweep of distances and reports the share of the object's own pixels that change,
split into the part whose outline moved and the part that merely shaded
differently. The two want different remedies: a moved outline needs triangles, a
changed shading needs a normal map baked from the fine mesh. `tools/lod_quality.py`
runs that sweep over a model and writes a contact sheet;
`tools/lod_transitions.py` draws each switch at the distance it would happen,
which is the frame a player would actually see.

## Or author them in Blender

Not every chain is baked from a scan. An artist with a model open in Blender
wants to cut the levels there and export them, and the add-on in
`src/OpenGLContext_editor/blender/openglcontext_lod` is that:

```bash
python -m OpenGLContext_editor.blender --package   # a zip Blender installs
```

Blender then has **Object > Make LOD chain**, which cuts a chain from the
selected mesh with Blender's own Decimate modifier and marks the levels, and a
glTF export extension that writes them as `MSFT_lod` — so the ordinary
*File > Export > glTF 2.0* produces a file a viewer switches levels in, and one
that has never heard of the extension draws the finest level.

The zip carries a `blender_manifest.toml` and installs as a Blender 4.2+
**extension**, with nothing of this toolkit in it. That is deliberate:
`MSFT_lod` is a Khronos vendor extension, and somebody exporting to three.js or
Babylon should not have to install a renderer to write one.
[docs/blender.md](docs/blender.md) is the guide.

The chain's last level does not have to be a mesh: `--impostor 8` bakes an
**octahedral impostor** — one view of the model per direction in a single
texture, on a card turned to the viewer. It lets a chain stop
decimating early: on the demo hall, four mesh levels and a card draw 62% fewer
triangles than six mesh levels, in one fewer draw call. Whether that is *faster*
depends on what the frame is waiting on — it bought nothing on a discrete GPU
and 2.4× on a software rasteriser; [docs/blender.md](docs/blender.md) has the
numbers.

The chain is cut to a **triangle budget** rather than to a ratio — *the finest
level is not to exceed twenty thousand* — because that is what an author has.
Blender's Decimate takes only a ratio and reports a read-only face count, so
`openglcontext_lod/budget.py` turns one into the other.

The demo world for the whole mechanism is built this way:

```bash
oglce-gallery --output gallery/gallery.glb
```

A hall of 120 marble busts on plinths, each a six-level chain, with a polished
parquet floor and dark beams overhead — CC0 art fetched from Poly Haven and
ambientCG, assembled and exported in Blender. It is what the engine ships as
its level-of-detail demo, so what a release carries is what this add-on makes
rather than a second path that might disagree with it.

## A reduced scan needs its own texture

A photogrammetry scan arrives unwrapped by the scanner, into thousands of small
charts. That is the right answer for the dense mesh — small charts distort least
— and the wrong one for anything reduced from it. A texture coordinate means
something only inside one chart, so once a reduced triangle covers more surface
than a chart holds, its three corners point at unrelated places in the image and
what it draws is the stripe between them. On a museum scan of lekking ruffs that
is the ground the birds stand on, and it starts at eight thousand triangles,
which is a level a game ships.

No decimator mends it: the fault is in the unwrap rather than the surface, and a
reducer can only carry the coordinates it was given. What mends it is the step
every scan pipeline takes — unwrap the reduced mesh afresh, then bake the
original into the new atlas:

```python
from OpenGLContext_editor.assets.rewrap import unwrap, bake

laid = unwrap(level_positions, level_indices)
image = bake(laid, source_positions, source_indices, source_uv, source_image)
```

`unwrap` welds the surface first — a mesh split by the atlas it arrived with is
not a torn surface, and an unwrapper handed it unwelded gives thousands of
charts back — then lays out charts made of whole triangles, so the failure above
cannot happen to the result at any triangle count. It says in `source` which of
the caller's vertices each new one came from, which is how normals, colours and
weights follow the vertex splits an unwrap makes.

`bake` fills that atlas by asking the original what it shows at each texel: the
place on the reduced surface is projected onto the nearest original triangle,
the original's own texture coordinate is read there, and the source image is
sampled. Charts are grown `BLEED` texels outwards so a renderer filtering at a
chart edge finds the chart rather than the black the atlas started as. The
result is bytes of the source's own channel count, `DEFAULT_SIZE` being 2048 to
a side.

Where a texel reads from depends on the two surfaces and nothing else, so a
model carrying base colour, roughness, occlusion and normals asks the same
question four times. `project` asks it once and `sample` spends the answer per
map:

```python
shot = project(laid, source_positions, source_indices, source_uv)
colour = sample(shot, base_colour_image)
rough = sample(shot, roughness_image)
```

A **tangent-space normal map** cannot be resampled that way. Each texel is a
direction relative to the frame the texture coordinates define at that point, so
a fresh unwrap turns the frame and the same bytes then mean a different
direction — lighting that leans the wrong way, with nothing odd about the image
to say so. `sample_normals` takes each texel the whole way round instead,
decoding against the original's frame and encoding against the new one:

```python
onto = frames(source_positions, source_indices, source_uv, source_normals)
into = frames(laid.positions, laid.indices, laid.uv, level_normals[laid.source])
bumps = sample_normals(shot, normal_image, onto, into, source_indices, laid.indices)
```

`frames` builds those from the engine's own `estimate_tangents`, per vertex and
interpolated across each face — the frame a fragment is given — so a map baked
here means what the shader reads. A per-face frame instead would step at every
triangle edge and the mesh would draw its own wireframe in the lighting.

This needs [xatlas](https://pypi.org/project/xatlas/), which is MIT and ships
wheels for CPython and PyPy alike. It is an extra, because it is a bake step and
nothing a game runs imports it:

```bash
pip install "OpenGLContext-editor[rewrap]"
```

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

## Detail the tiles carry

A world whose ground is meshed into its tiles (`ground='tiles'`) gets more out
of refining than a denser sampling of the same smooth function. Two things are
added as the tree descends, and both are bounded by what the level can show:

- **The grain in the ground.** `OpenGLContext.scenegraph.terrain.Relief` is a
  band of noise per feature size; a tile carries the bands its own samples can
  resolve, and the whole displacement is scaled to fit inside the tile's
  geometric error. `ProceduralWorld.grain` is which grain a world has, and
  `None` leaves the tiles smooth. Keep the features small: a band is as tall as
  its `roughness` times its own width, so a coarse band is a dune rather than a
  hummock, and what a landscape is *shaped* like is the height function's job.

  **How much detail can be felt is set by `field_resolution`.** The grain goes
  into the landscape, and the landscape is a grid: a band finer than it can
  hold is cut before anything draws it. At a sample every two metres that
  leaves one swell of about half a metre across sixteen — modulation across a
  hillside rather than hummocks and ruts.
- **Loose stone.** `bake.stones.StoneLayer` strews knee-high rock over the
  hillsides and writes what a tile can show as placements of a handful of
  shapes — one node per shape, however many stones the tile holds. A stone
  appears once the tile's error is within `DETAIL` times its radius, so a
  hillside fills in by size rather than switching on. `STONE_DENSITY`,
  `STONE_RADIUS` and `STONE_SLOPE_LIMIT` on the world are the knobs, and
  `MOST_PER_TILE` caps what one tile draws.

**The cleared corridor takes everything.** A road is a strip of ground that was
cleared to build it, and nothing a machine went through is still standing on it:
trees, boulders and loose stone are all kept off the same strip
(`ProceduralWorld.outside_the_clearing`, widened where a driver has to see round
a bend and again over a bore), each by its own size on top of it.

**What is drawn is what is collided against.** Both go into the landscape
written beside the tileset, not only into the pictures:

- the grain is added to the height function the landscape is sampled from, so
  the field a car is driven on, a camera is clamped to and a tree is planted on
  is the same surface the finest tile draws. A band the field's own grid cannot
  hold is cut before anything draws it (`Relief.no_finer_than`), so there is no
  relief a player can see and walk through. Ground the road *cleared* keeps
  none of it: `ProceduralWorld.grain_applies` fades it from 0 on the made
  ground to 1 beyond the corridor, because a hummock in the carriageway is one
  a grader took out;
- every stone travels in the tileset's `extras.stones` and a game stands the
  ones near it up as *domes* — a wheel rides over one, a walker stands on one,
  and a block the size of a stone would be a kerb across the hillside. Its own
  channel rather than the world's `props`, because a boulder has to stop a car
  from a long way off and a stone only has to be there where the wheel is.

**How much of it is seen is the tree's depth.** A world of `extent` metres
meshed at 33 samples a tile has a finest spacing of `extent / 2**depth / 32`, and
nothing finer than that is drawn. Two kilometres at `depth = 3` is a sample every
7.8 m, which carries none of the grain and none of the stone; at `depth = 5` it
is a sample every 2 m, which carries both. Each level is four times the tiles, so
it is a decision per world rather than a default.

The world is **told** that depth (`ProceduralWorld.depth`), because the same
number decides what a portal has to cover. A bore's mouth is cut on the drawn
ground's own grid, so the face has to be at least a cell wide or there is
daylight down each side of it, and the road's own space is cleared for a few
cells in front of it. Measured against the root tile instead of the finest, that
is a sixty-metre headwall and a trench down the approach.

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
from OpenGLContext.scenegraph.road import bank_profile
from OpenGLContext_editor.world.road import (
    RoadPath, RoadLayer, conform_terrain_at, follow_terrain,
)

course = follow_terrain(my_route, my_heights, spacing=5.0,
                        maximum_grade=0.075, design_speed=47.0,
                        minimum_height=2.5, closed=True)
path = RoadPath(course, bank=bank_profile(course, speed=47.0, closed=True))
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

`bank_profile` then says how far each corner **leans**. A superelevated corner
puts part of the car's weight to work holding it on the line, so it is faster
than the same corner flat — or as fast round a tighter radius, which is what
lets an alignment follow a valley instead of sweeping across it. Each corner
gets the lean that balances a car at the speed given, capped at `MAXIMUM_BANK`
(0.10 — the upper end of what is built into a road, not an oval's banking), and
the change from camber to full lean is spread over a transition of some seventy
metres on the approach rather than happening at the corner's entry. The lean
goes into the `RoadPath`, and from there into the surface, the earthwork beside
it, the structures carrying it and the tileset a game reads. `RoadPath()` with
no `bank` is a road whose corners are flat.

### One road, several kinds of road

`follow_terrain` takes `smoothing`, `maximum_grade` and `design_speed` as **one
figure or one per point of the alignment**, and that is what makes a road that
is not the same road all the way round.
`OpenGLContext_editor.world.character` works out what each of them should be:

```python
from OpenGLContext_editor.world.character import corner_radii, road_character
from OpenGLContext_editor.world.route import hold_corners

plan = hold_corners(my_route, corner_radii(my_route, design_radius=270.0,
                                           seed=3), closed=True)
kind = road_character(plan, my_heights, design_speed=200 / 3.6, spacing=6.0,
                      closed=True, climbing_lane=3.6)
course = follow_terrain(plan, my_heights, spacing=6.0, closed=True,
                        smoothing=kind.smoothing,
                        maximum_grade=kind.grade_limit,
                        design_speed=kind.design_speed)
```

Nothing there is sprinkled about; every figure is **derived** from the corner
the road is on, the land under it, or how far ahead a driver has to see:

| | |
|---|---|
| **the corners** | drawn from a mix (`CORNER_MIX`) so a lap has a hairpin and a sweeper as well as the corner it was laid out for. The plan's own turn angles are untouched — what changes is how hard each turn has to be taken |
| **how fast a stretch is for** | what its own corner allows, never more than the road's design speed. A crest inside a hairpin rounded for the speed of the straight before it is a quarter of a kilometre of earthwork for a crest nobody meets at that speed |
| **how steeply it may climb** | the ordinary limit, raised towards `steep_grade` where the land itself climbs harder than that over a quarter of a kilometre. Held to a gentle grade, a road across a hillside stands off it on an embankment for as far as the hillside lasts |
| **how much it is smoothed** | with the speed. A bump taken at two hundred is a car in the air and has to go; the same bump at eighty is the road having some shape, and ironing it out costs the drive and buys nothing |
| **how far the trees are cut back** | what a driver needs to see round the bend they are on. Sight round a bend of radius *r* past an obstruction *clear* to the side is about `sqrt(8 · r · clear)`, so the corridor that buys a stopping distance falls out of it. It is the corners **near the design radius** that get opened out: a tighter one is taken slowly enough to see round already, a wider one is straight enough |
| **where it is wide enough to be passed on** | the sustained climbs — 4% or more for four hundred metres — worst first, until `lane_share` of the road has been widened. Hill country asks for more climbing lanes than anybody builds, and a road widened along half its length is a wide road rather than a road with passing places on it |

`RoadCharacter.summary()` reports the range each covers, and `varies()` says
whether this came out as a road of one character after all.

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
else. Beside a banked corner the two verges are not at one height, and the
ground meets each of them where it actually is. A road already on the land disturbs almost nothing; one carried forty
metres over a valley builds an embankment as wide as it needs. Under the
carriageway the ground sits a hand's breadth below the surface, because a road
is built on a formation and surfaced on top of it.

A **portal is dug**. The cutting stops where a bore begins and the hillside
takes over, so left alone the ground steps from the carriageway to the hill
between one sample and the next — a face one sample thick with the arch cut out
of it. The ground around each portal is held down to the top of the portal's
face, with `PORTAL_SOIL` of ground over it, and rises from there at the same
batter the rest of the cutting uses, out to `PORTAL_CUT`. An ordinary hillside
is met well inside that and nothing is cut past where it is met; a hill too
steep to meet there is a hill rather than a doorway, and is left alone. The
funnel only ever takes ground away, so the cutting the road arrives in is
untouched and so is the hill the bore runs under.

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

**A drawn route keeps the corners it was drawn with.** `corner_radii` is for
the circuit `ProceduralWorld` invents for itself; a route a caller gives is held
to the one design radius, because its corners are already a decision somebody
made and re-drawing them from a mix is the generator overruling the designer.
What a drawn route still gets is everything `road_character` derives, all of
which follows from the line they drew rather than replacing it.

**How tight "too tight" is depends on how far the corner leans.**
`ProceduralWorld.corner_radius()` is the floor, and it is
`cornering_radius(design_speed, bank=maximum_bank)` with a twentieth over for
what draping and re-sampling move — 270 m for the shipped circuit, against 315 m
if the same speed had to be held flat. `ProceduralWorld(maximum_bank=0.0)` lays
the circuit out level and gets the wider corners back. A plan whose legs are too
short to fit the fillet keeps its corner and loses radius instead, and such a
corner is slower than the design speed: the road says so, since what it is
signed at comes from the radius and the lean it ended up with.

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
| `src/OpenGLContext_editor/assets/` | turning published art into assets: Poly Haven fetching, plant baking, billboards, and the unwrap-and-rebake for reduced scans |
| `src/OpenGLContext_editor/meshlod/` | levels of detail: one recorded reduction sliced into rungs, what each rung costs to look at, and the `MSFT_lod` glb they ship in |
| `src/OpenGLContext_editor/blender/` | the Blender add-on: LOD chains from the Decimate modifier, `MSFT_lod` on glTF export, and the bust gallery it builds |
| `src/OpenGLContext_editor/bin/` | `oglce-gallery`, which builds the demo world; and `oglc-bake`, which says the command is now `glisteel-bake` |
| `tools/` | authoring scripts run by hand: the level-of-detail quality sweep and the transition sheet |
| `docs/` | [authoring levels of detail in Blender](docs/blender.md) |
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
