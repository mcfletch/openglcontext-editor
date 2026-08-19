# Format specifications

Every format constant, layout and behaviour this package writes or reads cites a
numbered fact in one of these documents. They are the only channel through which
format knowledge reaches the code, and each one records where its own facts came
from.

| Spec | Covers |
|---|---|
| [ELEVATION-DATA.md](ELEVATION-DATA.md) | The SRTM `.hgt` height file's layout and naming, WGS 84, and the local tangent plane that puts a degree square on the ground in metres. |

## Where the facts are allowed to come from

The output formats are **published specifications**, which are permitted sources
and need no wall at all:

- **OGC 3D Tiles 1.1** — the tileset, its bounding volumes, geometric error and
  refinement.
- **glTF 2.0** (Khronos) and the `KHR_*` / `EXT_*` extension registry — tile
  content, materials, and `EXT_mesh_gpu_instancing` for baked instances.

Cite the specification, by section, in the code that relies on it. A constant
with no fact behind it means either a gap in these documents or a fact that came
from the wrong side of the wall.

## Where they are not

This package has two standing exposures to copyleft source, and both are
governed by [CLEAN-ROOM.md](CLEAN-ROOM.md):

- **GIS tooling.** GRASS and parts of the GDAL ecosystem are GPL. Coordinate
  reference systems, DEM formats and raster conventions are all documented
  independently of them; go to the documentation, or to the bytes of a sample
  file.
- **Road, terrain and world-building code in open game engines.** Most of it is
  GPL or AGPL. Nothing this package generates is required to match any of it,
  so the geometry here is derived from the mathematics and from what the engine
  renders, never from reading one.

A fact that cannot be found outside a copyleft source goes through the wall:
a Reader who writes no project code produces a spec in this directory, and the
implementer reads the spec and nothing else. If a fact cannot be separated from
its expression, escalate rather than paraphrase it.
