"""A world you can bake without authoring anything first.

The layers here describe a forested landscape -- hills, ridges, a river canyon
and a lake basin, with conifers on the ground that will hold them -- assembled
entirely from what the engine already generates. It is the worked example the
toolkit ships: ``oglc-bake`` bakes it, ``oglc-view`` streams the result, and the
code below is the shortest honest answer to "how do I describe a world?".

Every piece of it is a normal layer, so a world of your own is this file with
your height function, your assets and your scatter rules.
"""
from __future__ import annotations

from dataclasses import dataclass

from OpenGLContext.loaders.tiles3d import foliage
from OpenGLContext.loaders.tiles3d.procedural import (
    WATER_LEVEL,
    terrain_colors,
    terrain_height,
)
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

from OpenGLContext_editor.bake.assets import combined_mesh, meshes_from_gltf
from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.layers import HeightfieldLayer, InstanceLayer, Layer
from OpenGLContext_editor.world.scatter import scatter_on_heightfield, yaw_quaternions

#: Trees per square metre. A tenth of the forest demo's near-field density: this
#: world is 4 km across, and what a baked tile carries is the *sparse* layer of
#: real trees, with the near-field thicket left to the runtime.
TREE_DENSITY = 0.004

#: Where conifers will grow: above the waterline, below the snow, and off
#: anything a tree would slide down.
TREE_ELEVATION = (WATER_LEVEL + 2.0, 130.0)
TREE_SLOPE_LIMIT = 38.0

#: The tile error at which a conifer becomes two crossed cards. Roughly the
#: point at which a tree covers a few pixels, so the swap is not seen.
IMPOSTOR_ERROR = 8.0

CREDITS = (
    "Terrain, foliage textures and tree geometry: generated procedurally by "
    "OpenGLContext (BSD-3-Clause).",
)


@dataclass
class ProceduralWorld:
    """The knobs on the shipped example world.

    ``extent`` is the side of the square it covers, in metres, centred on the
    origin. ``resolution`` is how many ground samples across each tile gets --
    the vertex budget that, divided by the tile's size, sets the detail.
    """

    extent: float = 4096.0
    resolution: int = 33
    tree_density: float = TREE_DENSITY
    tree_height: float = 9.0
    seed: int = 11

    def footprint(self) -> BoundingBox:
        """The ground the world covers, as a footprint with no height.

        This is the terrain layer's extent. The region a *bake* partitions is
        taller than this -- it has to hold the hills and the trees standing on
        them -- and is left to the layers to report.
        """
        half = self.extent / 2.0
        return BoundingBox((-half, 0.0, -half), (half, 0.0, half))

    def layers(self) -> list[Layer]:
        """The world as a list of layers, ready for :func:`bake_world`."""
        return [self.terrain(), self.trees()]

    def terrain(self) -> HeightfieldLayer:
        return HeightfieldLayer(
            height_fn=terrain_height, extent=self.footprint(),
            resolution=self.resolution, color_fn=terrain_colors,
            water_level=WATER_LEVEL, name='terrain')

    def trees(self) -> InstanceLayer:
        scatter = scatter_on_heightfield(
            terrain_height, self.footprint(), density=self.tree_density,
            seed=self.seed, scale_range=(0.75, 1.35),
            slope_limit=TREE_SLOPE_LIMIT, height_range=TREE_ELEVATION)
        return InstanceLayer(
            positions=scatter.positions,
            rotations=yaw_quaternions(scatter.yaws),
            scales=scatter.scales,
            lods=[(0.0, conifer_mesh(self.tree_height, self.seed)),
                  (IMPOSTOR_ERROR, impostor_mesh(self.tree_height, self.seed))],
            name='trees')


def conifer_mesh(height: float = 9.0, seed: int = 11) -> PBRMesh:
    """The near rung: the engine's textured conifer, flattened to one mesh."""
    return combined_mesh(meshes_from_gltf(foliage.conifer_glb(height, seed=seed)))


def impostor_mesh(height: float = 9.0, seed: int = 11) -> PBRMesh:
    """The far rung: two crossed alpha cards, four triangles for a whole tree."""
    return combined_mesh(meshes_from_gltf(
        foliage.tree_billboard_glb(width=height * 0.65, height=height, seed=seed)))
