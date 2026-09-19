# Authoring levels of detail in Blender

A Blender add-on that gives Blender two things it does not have: a way to say
"these meshes are levels of detail of one another", and a glTF export that
writes that as `MSFT_lod` — the vendor extension the ecosystem uses and
[OpenGLContext reads](https://github.com/mcfletch/openglcontext/blob/main/docs/lod.html).

It needs only Blender's own Python and the glTF exporter Blender ships. An
author installs it into Blender and never installs this toolkit at all.

## Install it

`MSFT_lod` is a Khronos vendor extension rather than anything of this project's,
so the add-on installs into Blender the ordinary way and needs none of this
toolkit. From a release, or from a zip you build yourself:

```bash
python -m OpenGLContext_editor.blender --package        # writes the zip
```

Then in Blender, *Edit > Preferences > Get Extensions > Install from Disk*, or
from a command line:

```bash
blender --command extension install-file -r user_default -e openglcontext_lod-1.0.0.zip
```

It carries a `blender_manifest.toml`, so Blender 4.2 and later install it as an
**extension**; the `bl_info` beside it is the same add-on described for 4.0 and
4.1, which read that instead.

From a checkout, `python -m OpenGLContext_editor.blender --install` copies it
straight into the user add-on directory for whatever Blender is on the path,
which saves a step while working on it.

The directory is a complete add-on as it stands, so a checkout is enough:

```python
import sys, bpy
sys.path.insert(0, '.../src/OpenGLContext_editor/blender')
bpy.ops.preferences.addon_enable(module='openglcontext_lod')
```

Enabling it is what matters for export, not importing it: Blender's glTF
exporter looks for export extensions among the **enabled add-ons**, so a module
that is merely imported writes no levels — and the export would look like it
worked.

## Make a chain

Select a mesh and run **Object > Make LOD chain**, or the button in the *LOD*
tab of the 3D view's sidebar. It cuts the levels with Blender's own Decimate
modifier in collapse mode — which every Blender has and which interpolates the
UVs, so the levels keep the material they came with — names them
`<Name>_LOD1`, `<Name>_LOD2` … and marks each with the chain it belongs to.

| Option | What it does |
|---|---|
| Levels | How many levels the chain has, counting the original |
| Ratio | What share of the triangles each level keeps of the one before it |
| Finest level at most | A triangle budget for LOD0: a denser mesh is decimated down to it before the chain starts. 0 leaves the mesh as it came |
| Hide the coarse levels | Puts the alternatives out of the viewport. They are still exported — the export takes them out of the *scene* instead |

The ratio is measured against the original, which is what the modifier does:
asking for a half twice is a quarter of the first mesh, not a quarter of the
second.

**Budgets, not ratios.** Blender's Decimate modifier takes a `ratio` and
nothing else — its `face_count` is read-only and only reports what came out —
but an author has a budget: *the finest level is not to exceed twenty
thousand*. **Finest level at most** is that, and the rest of the chain follows
from it, so a 500,000-triangle scan and a 30,000-triangle game asset both start
their chains in the same place. The arithmetic is
`openglcontext_lod/budget.py`, which holds no Blender; a collapse removes whole
edges, so the result lands near the budget rather than exactly on it.

**Check before you export.** *Check LOD chains* reports what the scene would
write — how many chains, how many levels — and refuses the two mistakes that
produce a wrong file rather than an error: two objects claiming the same level
of one chain, and a chain whose thresholds get *larger* as it gets coarser.

## What marks a level

Custom properties on the object, which the operator sets and you can edit:

| Property | Means |
|---|---|
| `lod_group` | Which chain this is a level of. Objects sharing one are one chain |
| `lod_level` | How fine: `0` is the finest. It **orders** the chain rather than indexing it, so deleting the middle of a chain leaves a shorter chain, not a hole |
| `lod_coverage` | Optional. The share of the window's *height* at which this level takes over |

A name in the `Bust_LOD2` form is read where the properties are absent — so a
chain brought in from elsewhere needs nothing done to it — and the properties
win where both are present. Blender's own `Bust.001` duplicate suffix is not
mistaken for a level.

### Thresholds

`lod_coverage` is a share of the window's height, decreasing down the chain.
State one on **every** level of a chain or on none: half a series is not a
series, and filling the gap by halving would bury the author's intent in a
number that looks deliberate. A chain that states none is written with a halving
series — a half, a quarter, an eighth — ending at zero, so a level nobody
measured a threshold for is never the reason something disappears.

Thresholds derived from *measurement* are better than a series, and
`OpenGLContext_editor.meshlod`'s `measure_chain` is where they come from: it
draws each level against the original over a sweep of distances and reports the
share of pixels that change. See the level-of-detail section of the
[README](../README.md).

## The last level does not have to be a mesh

Past a certain distance a mesh is the wrong thing entirely: a bust twenty pixels
tall spends five hundred triangles on a silhouette a picture would draw exactly.
An **octahedral impostor** is that picture — one view of the model per
direction, baked into a single square texture, drawn on a card turned to the
viewer that shows whichever view matches where they are standing.

```python
from openglcontext_lod import impostor

baked = impostor.bake_atlas(bust, '/tmp/bust_impostor.png', grid=8, image=256)
card  = impostor.impostor_mesh(baked, 'bust_LOD4')   # a level like any other
```

`bake_atlas` renders `grid × grid` views of the object in Blender — EEVEE,
orthographic, on a transparent film — and lays them out by the octahedral fold
in `octahedral.py`. The scene's render settings, camera and world are put back
afterwards, so it can be run on the file you are working in.

**How big to make it** is a question about how small the model will be on screen
when the impostor takes over, and nothing else. At a switching threshold of
three per cent of a 720-line window the model is twenty-odd pixels tall, so 256
pixels at 8 views a side — 32 pixels a view — covers it. Twice as many views is
four times the texture for angles a distant object does not resolve.

**It lets a chain stop decimating early**: four mesh levels and a card carry a
model further than six mesh levels do. On the demo hall with 108 busts on
screen that is 62% of the triangles and one draw call fewer — 39,460 against
103,536, in 10 draws against 11.

**Bake one only when geometry is what a frame is waiting on.** On a
discrete-class GPU that change made the gallery *no faster at all*: 4.93 ms a
frame against 4.96, which is noise, because the frame was spending its time on
processor work per object rather than on vertices. The same two worlds on a
software rasteriser — which is roughly how a weak integrated part behaves —
went **46.8 ms to 19.1 ms, 2.4× faster**. Same change, nothing on one machine
and a different game on the other. Measure which yours is before baking
anything.

Two things the bake settles that are easy to get wrong, and that this does for
you: Blender views a render through **AgX** by default, and a card baked through
a film curve is a paler, flatter version of the model at the moment it appears —
the bake pins `Standard`. And Blender has had no **alpha-clip** blend mode since
4.2, so a cut-out material exports as `BLEND` and is drawn as glass; the export
hook writes `MASK`, which is what keeps impostors in the opaque batch.

What it does not do: it shows its nearest view rather than a blend of the
nearest few, so turning past the angle between two baked views swaps one picture
for another; and it carries the lighting it was baked under — an even white
surround — rather than the lighting around it.

## Export

**File > Export > glTF 2.0** as normal. The add-on's export extension turns
every chain in the scene into `MSFT_lod` on the way out; there is nothing to
switch on per export. *File > Export > glTF 2.0 with MSFT_lod (.glb)* is the
same export with the settings a world wants already set — cameras and lights
kept, and no limit to what is visible, because the coarse levels are hidden.

What lands in the file:

```json
"nodes": [
    {"name": "Bust_LOD0", "mesh": 0,
     "extensions": {"MSFT_lod": {"ids": [1, 2]}},
     "extras": {"MSFT_screencoverage": [0.5, 0.25, 0.0]}},
    {"name": "Bust_LOD1", "mesh": 1},
    {"name": "Bust_LOD2", "mesh": 2}
]
```

The finest level carries the extension, `ids` names the coarser ones in
decreasing detail, and the alternatives are taken out of the scene — a level
still reachable from a scene is a level still drawn, which would put the whole
chain on screen at once in one place. A reader that has never heard of the
extension draws the finest level, which is the correct thing for it to do.

**Share the mesh data between copies.** A linked duplicate (`Alt+D`) exports as
another node over the same mesh, and copies sharing a mesh are what let a
renderer collapse them into one instanced draw. A full copy (`Shift+D`) exports
a second mesh, and nothing can batch it with the first.

## The gallery

The demo world is built by this add-on, driven headlessly:

```bash
oglce-gallery --output gallery/gallery.glb --blend gallery/gallery.blend
oglce-gallery --output gallery/gallery.glb --levels 4 --impostor 8
```

That fetches the CC0 art — the bust from Poly Haven, the surfaces from ambientCG
— builds the hall in Blender and exports the glB, writing `CREDITS.txt` beside
it from what was actually fetched. `--content-only` stops after the fetch,
`--bays` changes how long the hall is, and `--blend` saves the Blender file,
which is the one to open to see how the world is put together.

Everything that decides *where* things go is in
`openglcontext_lod/gallery.py` and holds no Blender, so the arrangement can be
checked without opening a window; `scene.py` is the part that calls `bpy`.

## Where it is tested

The parts that carry no Blender — what a chain becomes in a document, where the
gallery puts its walls — are tested in this project's own suite, under Python
3.12, with no Blender involved. The parts that do are tested by running Blender
headless: `tests/test_blender_export.py` builds a scene, exports it and reads
the JSON back. Those cases are marked `blender` and are skipped where no Blender
is on the path.
