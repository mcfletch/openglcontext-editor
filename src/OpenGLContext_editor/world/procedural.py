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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
from OpenGLContext_editor.world.road import (
    RoadLayer,
    RoadPath,
    conform_terrain,
    conform_terrain_at,
    follow_terrain,
)
from OpenGLContext_editor.world.scatter import scatter_on_heightfield, yaw_quaternions
from OpenGLContext_editor.world.structures import Op, choose_structures

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

#: The circuit: an oval bent out of round by two harmonics, so it has long
#: straights, a pair of fast sweepers and one slower complex, rather than being
#: a ring the car can hold at full throttle.
CIRCUIT_HARMONICS = ((3, 0.20), (5, 0.09))

#: The alignment is smoothed over this many metres of road before it is built,
#: so the track carries the shape of the landscape without its every hummock,
#: and no grade steeper than this fraction survives.
CIRCUIT_SMOOTHING = 90.0
CIRCUIT_MAX_GRADE = 0.075

#: How fast the circuit is meant to be driven, in metres per second (170 km/h),
#: which is what rounds off its crests: a change of grade sharp enough to take a
#: car's wheels off the road at this speed is spread into a vertical curve that
#: does not.
CIRCUIT_DESIGN_SPEED = 47.0

#: The circuit stays this far above the waterline. Where the ground is lower --
#: the lake basin, the floor of the canyon -- the road rides over it on fill and
#: its approaches climb to meet it.
CAUSEWAY_FREEBOARD = 2.5

#: No tree stands closer to the road than its own half-width plus this, in
#: metres -- the cleared corridor a road is built inside.
ROAD_CLEARANCE = 6.0

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

    ``route`` is the circuit's plan as an (N,2) array of XZ points -- what a
    designer drew. Left out, the world draws its own. Either way it is a
    *plan*: it arrives with no heights on it, and everything else about
    assembling the world is the same, which is the point of it being one
    argument rather than a second class.

    ``structures`` decides whether the alignment's large departures from the
    land are built as bridges and tunnels. With it off the same road is carried
    entirely on earthworks, which over a landscape of this relief means
    embankments and cuttings the size of the hills they cross.
    """

    extent: float = 4096.0
    resolution: int = 33
    tree_density: float = TREE_DENSITY
    tree_height: float = 9.0
    seed: int = 11
    road: bool = True
    #: The circuit's plan, (N,2) XZ; None for the world's own.
    route: Any = None
    #: Whether the route returns to where it started.
    closed: bool = True
    #: Whether a deck or a bore is built where the earthworks would be huge.
    structures: bool = True
    wetness: float = 0.0
    _circuit: RoadPath | None = field(default=None, init=False, repr=False)

    def footprint(self) -> BoundingBox:
        """The ground the world covers, as a footprint with no height.

        This is the terrain layer's extent. The region a *bake* partitions is
        taller than this -- it has to hold the hills and the trees standing on
        them -- and is left to the layers to report.
        """
        half = self.extent / 2.0
        return BoundingBox((-half, 0.0, -half), (half, 0.0, half))

    def layers(self) -> list[Layer]:
        """The world as a list of layers, ready for :func:`bake_world`.

        The order the pieces are decided in matters: the circuit is laid out on
        the natural ground, the ground is then reshaped to meet it, and the
        trees are scattered over the *reshaped* ground with the road's corridor
        kept clear. A tree placed before the earthworks would stand in a cutting
        with its roots in the air.
        """
        layers: list[Layer] = [self.terrain(), self.trees()]
        if self.road:
            layers.append(self.circuit_layer())
        return layers

    def height_fn(self) -> Any:
        """The ground as the world finally has it, earthworks included."""
        if not self.road:
            return terrain_height
        return conform_terrain(terrain_height, self.circuit())

    def circuit(self) -> RoadPath:
        """The race circuit: laid out on the natural ground, smoothed, and told
        which of its stretches are carried rather than laid."""
        if self._circuit is None:
            plan = (np.asarray(self.route, dtype='d') if self.route is not None
                    else circuit_plan(self.extent * 0.36, self.extent * 0.28))
            line = follow_terrain(plan, terrain_height, spacing=6.0,
                                  smoothing=CIRCUIT_SMOOTHING,
                                  maximum_grade=CIRCUIT_MAX_GRADE,
                                  design_speed=CIRCUIT_DESIGN_SPEED,
                                  minimum_height=WATER_LEVEL + CAUSEWAY_FREEBOARD,
                                  closed=self.closed)
            self._circuit = RoadPath(line, ops=self._ops(line))
        return self._circuit

    def _ops(self, line: np.ndarray) -> Any:
        """What is built along the alignment, point by point."""
        if not self.structures:
            return None
        natural = np.asarray(terrain_height(line[:, 0], line[:, 2]), dtype='d')
        chosen = choose_structures(line, natural, waterline=WATER_LEVEL,
                                   closed=self.closed)
        ops = np.full(len(line), Op.DIRT, dtype=object)
        for structure in chosen:
            ops[structure.indices(len(line))] = structure.kind
        return ops

    def circuit_layer(self) -> RoadLayer:
        return RoadLayer(self.circuit(), wetness=self.wetness,
                         ground=terrain_height)

    def terrain(self) -> HeightfieldLayer:
        return HeightfieldLayer(
            height_fn=self.height_fn(), height_fn_at=self.height_fn_at(),
            extent=self.footprint(), resolution=self.resolution,
            color_fn=terrain_colors, water_level=WATER_LEVEL, name='terrain')

    def height_fn_at(self) -> Any:
        """The ground as a function of the spacing a tile samples it at.

        A cutting narrower than a coarse tile's vertex spacing would be stepped
        over and the road in it buried, so each tile gets the earthwork widened
        to its own resolution.
        """
        if not self.road:
            return None
        return conform_terrain_at(terrain_height, self.circuit())

    def trees(self) -> InstanceLayer:
        ground = self.height_fn()
        scatter = scatter_on_heightfield(
            ground, self.footprint(), density=self.tree_density,
            seed=self.seed, scale_range=(0.75, 1.35),
            slope_limit=TREE_SLOPE_LIMIT, height_range=TREE_ELEVATION,
            keep=self._away_from_the_road)
        return InstanceLayer(
            positions=scatter.positions,
            rotations=yaw_quaternions(scatter.yaws),
            scales=scatter.scales,
            lods=[(0.0, conifer_mesh(self.tree_height, self.seed)),
                  (IMPOSTOR_ERROR, impostor_mesh(self.tree_height, self.seed))],
            name='trees')

    def _away_from_the_road(self, points: np.ndarray) -> np.ndarray:
        """Which placements are outside the road's cleared corridor."""
        if not self.road:
            return np.ones(len(points), dtype=bool)
        circuit = self.circuit()
        corridor = circuit.profile.total_width / 2.0 + ROAD_CLEARANCE
        distance, _ = circuit.nearest(points[:, 0], points[:, 2],
                                      radius=corridor * 1.5)
        return np.asarray(distance > corridor)


def circuit_plan(radius_x: float, radius_z: float, points: int = 360,
                 harmonics: Sequence[tuple[int, float]] = CIRCUIT_HARMONICS
                 ) -> np.ndarray:
    """A closed race circuit as an (N,2) plan, smooth by construction.

    An ellipse whose radius is modulated by a few harmonics of the angle: the
    result has straights, sweepers and a slow complex, and -- being a sum of
    sinusoids -- has no corner anywhere for the alignment to have to round off.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    radius = np.ones_like(angle)
    for order, amount in harmonics:
        radius = radius + amount * np.sin(order * angle + order)
    return np.stack([radius_x * radius * np.cos(angle),
                     radius_z * radius * np.sin(angle)], axis=-1)


def conifer_mesh(height: float = 9.0, seed: int = 11) -> list[PBRMesh]:
    """The near rung: the engine's textured conifer.

    Two meshes, not one: the trunk wears bark and the needle skirts wear an
    alpha-masked needle card, and a tree of one material is a tree with bark
    for foliage.
    """
    return meshes_from_gltf(foliage.conifer_glb(height, seed=seed))


def impostor_mesh(height: float = 9.0, seed: int = 11) -> PBRMesh:
    """The far rung: two crossed alpha cards, four triangles for a whole tree."""
    return combined_mesh(meshes_from_gltf(
        foliage.tree_billboard_glb(width=height * 0.65, height=height, seed=seed)))
