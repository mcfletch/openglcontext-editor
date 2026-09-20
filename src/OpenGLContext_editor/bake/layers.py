"""What a tile's content is made of.

A *layer* answers one question for the bake: **what is in this region, at this
geometric error?** The driver walks the octree and asks every layer at every
node; what comes back is a list of glTF scene nodes, which the driver writes as
that tile's content.

Answering per region and per error is what makes level of detail work. The tree
refines with ``REPLACE``, so a node's content stands in for its whole subtree:
terrain is sampled at a fixed vertex count over a shrinking footprint (finer
ground the deeper you go), and instances are thinned to a budget so a coarse
tile carries a sparse stand-in for the dense scatter beneath it.

Four layers cover the first world:

:class:`HeightfieldLayer`   the ground, sampled from a height function
:class:`WaterLayer`         open water, wherever the ground dips below a line
:class:`InstanceLayer`      one mesh placed many times -- trees, props, rocks
:class:`MeshLayer`          meshes placed once, at a fixed position

A layer is anything with ``bounds()`` and ``content(region, error)``; the
:class:`Layer` protocol states that and nothing else, so a world can add its own.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from OpenGLContext.loaders.gltf.writer import InstanceSet, SceneNode
from OpenGLContext.loaders.tiles3d.procedural import terrain_patch
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.terrain import Relief
from OpenGLContext.scenegraph.terrain.ground import GROUND_MATERIAL
from OpenGLContext.scenegraph.terrain.holes import cut
from OpenGLContext.scenegraph.water import LAKE, mesh_across, water_surface

from OpenGLContext_editor.bake.bounds import BoundingBox

HeightFn = Callable[[Any, Any], Any]
#: ``holes(x, z) -> mask``: where a surface is not there.
Holes = Callable[[Any, Any], Any]
ColorFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: How many samples across a tile edge the ground is meshed at. 33 gives a
#: 32-quad patch, which is the resolution the engine's own terrain bakers use
#: and lands a tile comfortably inside a 16-bit index buffer.
DEFAULT_RESOLUTION = 33

#: The margin, in vertex spacings, by which a region's vertical test is
#: loosened. A node whose bottom face grazes the surface still holds ground.
VERTICAL_TOLERANCE = 1e-3


@runtime_checkable
class Layer(Protocol):
    """What the bake driver asks of anything it bakes.

    Two further methods are asked for *if a layer has them* -- they say nothing
    about the common case, so requiring them of every layer would make the
    simplest layer the one with the most to write:

    ``assets() -> {filename: bytes}``
        Files this layer's content refers to, by name relative to the tileset. A
        texture every tile uses is written once beside the tileset and named by
        each of them, rather than embedded in every one.

    ``metadata() -> dict``
        What a *game* needs to know about this layer, merged into the tileset's
        ``extras``. A baked world is more than what it looks like: where the
        road runs is not recoverable from a pile of triangles, but a game needs
        it to put a car on the track, time a lap, or drive an opponent round --
        so the layer that knows says so, once, and the world carries the answer.
    """

    name: str

    def bounds(self) -> BoundingBox | None:
        """Everything this layer could contribute, or None if it has nothing."""

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        """This layer's contribution to one tile, at that tile's error."""


# --- the ground ---------------------------------------------------------------

@dataclass
class HeightfieldLayer:
    """Ground meshed from a height function, one patch per tile.

    ``extent`` is the footprint the surface covers; its Y is ignored, since the
    height function decides that. Every tile is meshed at ``resolution`` samples
    across, so the ground gets finer as the tree descends without the tile's
    vertex count changing.

    ``skirt`` drops a vertical curtain around each patch, measured in vertex
    spacings, so the seam between a coarse tile and the finer ones beside it
    shows no gap. ``water_level`` clamps the surface flat at that height.

    ``height_fn_at`` is for ground whose shape depends on how finely it is
    sampled. A road cut into a hillside is the case that needs it: a cutting
    narrower than a coarse tile's vertex spacing is stepped straight over, and
    the road inside it disappears under the ground. Given
    ``height_fn_at(spacing) -> height function``, each tile asks for the ground
    at its own spacing and gets an earthwork it can actually represent. When it
    is set, ``height_fn`` is still what the layer reports its *bounds* from.

    ``holes`` is ground that is not there -- ``holes(x, z) -> mask``, the mouth
    of a bore the road runs into. The tile is meshed and then cut back to the
    opening's own edge (:func:`OpenGLContext.scenegraph.terrain.holes.cut`), so
    what the hole is shaped like is the portal rather than the tile's grid, and
    a tile the opening swallows whole carries no ground at all.

    ``relief`` is the grain in the ground -- the hummocks and ruts a height
    function does not carry -- meshed into a tile once its samples are close
    enough together to show it, and held inside the tile's own geometric error
    (:class:`~OpenGLContext.scenegraph.terrain.Relief`). What a coarse tile
    draws is unchanged by it, and no tile's surface stands further from the
    height function than the error the streamer already allows -- which is what
    lets a world be collided against the function while it is drawn from the
    tiles.
    """

    height_fn: HeightFn
    extent: BoundingBox
    resolution: int = DEFAULT_RESOLUTION
    height_fn_at: Callable[[float], HeightFn] | None = None
    color_fn: ColorFn | None = None
    water_level: float | None = None
    material: PBRMaterial | None = None
    holes: Holes | None = None
    relief: Relief | None = None
    skirt: float = 2.0
    name: str = 'terrain'

    def bounds(self) -> BoundingBox:
        """The extent's footprint, with the height range the surface reaches."""
        low, high = self._height_range(self.extent, samples=33)
        return self.extent.with_height(low, high)

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        footprint = self.footprint_of(region)
        if footprint is None:
            return []
        low, high = self._height_range(footprint, samples=9)
        depth = self._skirt_depth(footprint)
        # A node the surface passes nowhere near holds no ground. The skirt hangs
        # below the surface, so the test reaches that far down too.
        if (region.maximum[1] < low - VERTICAL_TOLERANCE
                or region.minimum[1] > high + depth + VERTICAL_TOLERANCE):
            return []
        positions, normals, colors, indices = terrain_patch(
            float(footprint.minimum[0]), float(footprint.maximum[0]),
            float(footprint.minimum[2]), float(footprint.maximum[2]),
            self.resolution, height_fn=self.height_fn_for(footprint, error),
            skirt_depth=depth, water_level=self.water_level,
            color_fn=self.color_fn)
        if self.holes is not None:
            opened = _opened(positions, normals, colors, indices, self.holes)
            if opened is None:
                return []
            positions, normals, colors, indices = opened
        mesh = PBRMesh(positions=positions, normals=normals, colors=colors,
                       indices=indices, material=self.material or _ground_material())
        return [SceneNode(mesh=mesh, name='%s_%d' % (self.name, self.resolution))]

    def sample_spacing(self, footprint: BoundingBox) -> float:
        """How far apart this tile's ground samples are, in metres."""
        return float(max(footprint.size[0], footprint.size[2])) / self.resolution

    def height_fn_for(self, footprint: BoundingBox, error: float) -> HeightFn:
        """The ground this tile is meshed from: the surface it *draws*.

        The height function with the tile's own earthwork resolution and its own
        relief already in it, so anything that has to sit on the drawn ground --
        a stone lying on the hillside -- asks the same question the mesh did.
        """
        spacing = self.sample_spacing(footprint)
        ground = (self.height_fn if self.height_fn_at is None
                  else self.height_fn_at(spacing))
        if self.relief is None:
            return ground
        grained: HeightFn = self.relief.over(ground, spacing=spacing,
                                             error=float(error))
        return grained

    def footprint_of(self, region: BoundingBox) -> BoundingBox | None:
        """The region's XZ overlap with the extent, or None if they miss."""
        low = np.maximum(region.minimum, self.extent.minimum)
        high = np.minimum(region.maximum, self.extent.maximum)
        if low[0] >= high[0] or low[2] >= high[2]:
            return None
        return BoundingBox((low[0], region.minimum[1], low[2]),
                           (high[0], region.maximum[1], high[2]))

    def _height_range(self, footprint: BoundingBox, samples: int) -> tuple[float, float]:
        xs = np.linspace(footprint.minimum[0], footprint.maximum[0], samples)
        zs = np.linspace(footprint.minimum[2], footprint.maximum[2], samples)
        gx, gz = np.meshgrid(xs, zs, indexing='ij')
        heights = np.asarray(self.height_fn(gx, gz), dtype='d')
        if self.water_level is not None:
            heights = np.maximum(heights, self.water_level)
        return float(heights.min()), float(heights.max())

    def _skirt_depth(self, footprint: BoundingBox) -> float:
        if self.skirt <= 0:
            return 0.0
        spacing = float(max(footprint.size[0], footprint.size[2])) / self.resolution
        return spacing * self.skirt


def _opened(positions: np.ndarray, normals: np.ndarray, colors: np.ndarray,
            indices: np.ndarray, holes: Holes) -> tuple[np.ndarray, ...] | None:
    """One patch cut back to the edge of the openings in it.

    The attributes travel through the cut with the positions, so a corner the
    cut invented is shaded and coloured as the surface already was there. What
    comes back is compacted, since a tile most of an opening swallowed would
    otherwise carry a grid of vertices no triangle names.
    """
    width = colors.shape[1]
    packed = np.hstack([np.asarray(positions, 'd'), np.asarray(normals, 'd'),
                        np.asarray(colors, 'd')])
    packed, triangles = cut(packed, np.asarray(indices).reshape(-1, 3), holes)
    if not len(triangles):
        return None
    kept, renumbered = np.unique(triangles, return_inverse=True)
    packed = packed[kept]
    return (packed[:, :3].astype('f'), packed[:, 3:6].astype('f'),
            packed[:, 6:6 + width].astype('f'),
            renumbered.reshape(-1).astype(np.uint32))


def _ground_material() -> PBRMaterial:
    """The material a tile's ground carries: vertex-coloured, two-sided.

    Two-sided because a tile's skirt is seen from both faces. Vertex-coloured
    because that is what any renderer makes of it, and **named**, because the
    engine's own makes more: a primitive whose material is called ``ground`` is
    drawn with the world's ground shading -- the detail materials blended per
    pixel from the control map beside the tileset, and the light baked into the
    landscape (:mod:`OpenGLContext.scenegraph.terrain.ground`). The colours stay
    on it either way, so a viewer that has never heard of that convention still
    draws a landscape rather than a white one.
    """
    return PBRMaterial(baseColor=(1.0, 1.0, 1.0), metallic=0.0, roughness=1.0,
                       doubleSided=True, DEF=GROUND_MATERIAL)


# --- open water ---------------------------------------------------------------

@dataclass
class WaterLayer:
    """Open water, wherever the ground dips below ``level``.

    Its own surface rather than the ground clamped flat at the waterline, which
    is what gives a world a **shoreline**: the shore is the line where the land
    passes through the surface, so the terrain has to be meshed as it is --
    dipping under -- and the water laid over it. A clamped ground has no such
    line to draw.

    A tile whose ground never reaches the waterline holds no water. One that
    dips below it anywhere is covered edge to edge, because a lake does not stop
    halfway across a tile.

    ``probe`` is how finely the ground is sampled to decide that, and
    ``resolution`` how finely the sheet itself is meshed -- which is not about
    its shape, since it is a plane, but about how finely the ripple in its
    normals is carried.
    """

    height_fn: HeightFn
    extent: BoundingBox
    level: float = 0.0
    #: How many vertices across the sheet is meshed at, or None to take it from
    #: the wave it is carrying (:func:`~OpenGLContext.scenegraph.water.mesh_across`).
    #: A fixed count over a tile hundreds of metres wide samples the ripple every
    #: few hundred metres, which aliases the wave away and leaves a flat plate.
    resolution: int | None = None
    probe: int = 17
    material: PBRMaterial | None = None
    #: How the surface moves. Open water with nowhere to go is a lake: still
    #: water is a mirror, and a mirror that size with nothing over it but a pale
    #: sky is a white plate lying in the landscape.
    style: Any = LAKE
    name: str = 'water'

    def bounds(self) -> BoundingBox:
        """The extent's footprint, at the waterline. Water has no thickness."""
        return self.extent.with_height(float(self.level), float(self.level))

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        footprint = self._footprint(region)
        if footprint is None:
            return []
        if not (region.minimum[1] <= self.level <= region.maximum[1]):
            return []
        if not self._flooded(footprint):
            return []
        side = max(float(footprint.maximum[0] - footprint.minimum[0]),
                   float(footprint.maximum[2] - footprint.minimum[2]))
        across = (int(self.resolution) if self.resolution is not None
                  else mesh_across(side, self.style))
        sheet = water_surface(
            float(footprint.minimum[0]), float(footprint.maximum[0]),
            float(footprint.minimum[2]), float(footprint.maximum[2]),
            level=float(self.level), resolution=across, style=self.style,
            material=self.material)
        return [SceneNode(mesh=sheet, name='%s_%d' % (self.name, across))]

    def _flooded(self, footprint: BoundingBox) -> bool:
        """Whether the ground under this footprint goes under the waterline."""
        xs = np.linspace(footprint.minimum[0], footprint.maximum[0], self.probe)
        zs = np.linspace(footprint.minimum[2], footprint.maximum[2], self.probe)
        gx, gz = np.meshgrid(xs, zs, indexing='ij')
        return bool(np.asarray(self.height_fn(gx, gz), dtype='d').min()
                    < self.level)

    def _footprint(self, region: BoundingBox) -> BoundingBox | None:
        """The region's XZ overlap with the extent, or None if they miss."""
        low = np.maximum(region.minimum, self.extent.minimum)
        high = np.minimum(region.maximum, self.extent.maximum)
        if low[0] >= high[0] or low[2] >= high[2]:
            return None
        return BoundingBox((low[0], region.minimum[1], low[2]),
                           (high[0], region.maximum[1], high[2]))


# --- placed copies of one mesh ------------------------------------------------

def _as_meshes(rung: Any) -> list[Any]:
    """A rung of a detail ladder, as the list of meshes it is made of."""
    if isinstance(rung, (list, tuple)):
        return list(rung)
    return [rung]


@dataclass
class InstanceLayer:
    """One mesh placed many times, written as ``EXT_mesh_gpu_instancing``.

    ``lods`` is the detail ladder: pairs of (the coarsest error this rung is good
    enough for, the rung), finest first. A tile picks the coarsest entry whose
    threshold it reaches, so distant tiles carry impostors and near ones carry
    the real geometry.

    A rung is one mesh, or several when the prototype needs more than one
    material -- a tree is a bark trunk and alpha-masked needles, and merging
    those into one mesh would paint the needles in bark. Each mesh is written
    as its own node over the same placements.

    ``max_instances`` caps what one tile writes. A coarse tile over a whole
    forest is thinned to that many by an even stride -- deterministically, so a
    re-bake produces the same world -- and the tiles beneath it carry the rest.
    """

    positions: np.ndarray
    lods: Sequence[tuple[float, Any]]
    rotations: np.ndarray | None = None
    scales: np.ndarray | None = None
    max_instances: int = 4096
    name: str = 'instances'

    def __post_init__(self) -> None:
        if not self.lods:
            raise ValueError("an instance layer needs at least one level of detail")
        self.positions = np.asarray(self.positions, dtype='d').reshape(-1, 3)
        self._ladder = sorted(((float(error), _as_meshes(rung))
                               for error, rung in self.lods),
                              key=lambda entry: entry[0])
        if self.scales is not None:
            scales = np.asarray(self.scales, dtype='f')
            if scales.ndim == 1:
                scales = np.repeat(scales[:, None], 3, axis=1)
            self.scales = scales

    def bounds(self) -> BoundingBox | None:
        return BoundingBox.of_points(self.positions)

    def mesh_for(self, error: float) -> list[Any]:
        """The meshes making up the rung a tile of this error draws."""
        chosen = self._ladder[0][1]
        for threshold, meshes in self._ladder:
            if error >= threshold:
                chosen = meshes
        return chosen

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        inside = np.nonzero(
            np.all((self.positions >= region.minimum)
                   & (self.positions <= region.maximum), axis=1))[0]
        if not len(inside):
            return []
        if len(inside) > self.max_instances:
            # An even stride rather than a random sample: the thinned set is the
            # same every bake, and it stays spread over the tile.
            stride = int(np.ceil(len(inside) / self.max_instances))
            inside = inside[::stride][:self.max_instances]
        instances = InstanceSet(
            translations=self.positions[inside].astype('f'),
            rotations=(None if self.rotations is None
                       else np.asarray(self.rotations, 'f')[inside]),
            scales=None if self.scales is None else self.scales[inside])
        return [SceneNode(mesh=mesh, instances=instances, name=self.name)
                for mesh in self.mesh_for(error)]


# --- meshes placed once -------------------------------------------------------

@dataclass
class MeshLayer:
    """Meshes at fixed positions -- a building, a bridge deck, a prop.

    A node is written into every tile whose region its bounds meet, so it is
    present at whatever detail the viewer has refined to. ``maximum_error``
    holds it back from tiles coarser than that, for content too small to be
    worth a distant tile's bandwidth.
    """

    nodes: Sequence[SceneNode]
    maximum_error: float | None = None
    name: str = 'meshes'
    _bounds: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._bounds = [node_bounds(node) for node in self.nodes]

    def bounds(self) -> BoundingBox | None:
        return BoundingBox.joined(self._bounds)

    def content(self, region: BoundingBox, error: float) -> list[SceneNode]:
        if self.maximum_error is not None and error > self.maximum_error:
            return []
        return [node for node, box in zip(self.nodes, self._bounds, strict=True)
                if box is not None and _overlaps(box, region)]


def _overlaps(a: BoundingBox, b: BoundingBox) -> bool:
    return bool(np.all(a.minimum <= b.maximum) and np.all(a.maximum >= b.minimum))


def node_bounds(node: SceneNode) -> BoundingBox | None:
    """A scene node's world bounds, instances included.

    The node's own scale and translation are applied to its meshes' points, and
    an instanced node is measured over every placement. Rotation -- the node's
    or an instance's -- is absorbed by widening the box to the sphere that
    encloses it, since the alternative is a box that a turned mesh sticks out
    of, and a tile whose content escapes its bounding volume is culled while it
    is still on screen.
    """
    meshes = ([node.mesh] if isinstance(node.mesh, PBRMesh)
              else list(node.mesh or ()))
    box = BoundingBox.joined(BoundingBox.of_points(mesh.positions)
                             for mesh in meshes if mesh.positions is not None)
    if box is None:
        return None
    if node.instances is not None:
        box = _instanced_bounds(box, node.instances)
    else:
        if node.rotation is not None:
            box = _rotation_proof(box)
        box = _placed(box, node.scale, node.translation)
    return box


def _placed(box: BoundingBox, scale: Any, translation: Any) -> BoundingBox:
    minimum, maximum = box.minimum, box.maximum
    if scale is not None:
        factor = np.asarray(scale, dtype='d')
        minimum, maximum = (np.minimum(minimum * factor, maximum * factor),
                            np.maximum(minimum * factor, maximum * factor))
    if translation is not None:
        offset = np.asarray(translation, dtype='d')
        minimum, maximum = minimum + offset, maximum + offset
    return BoundingBox(minimum, maximum)


def _rotation_proof(box: BoundingBox) -> BoundingBox:
    """The box that encloses this one however it is turned about its centre."""
    radius = float(np.linalg.norm(box.half))
    return BoundingBox.centred(box.center, (radius, radius, radius))


def _instanced_bounds(box: BoundingBox, instances: InstanceSet) -> BoundingBox:
    """The box enclosing one mesh placed at every instance of a set."""
    if instances.rotations is not None:
        box = _rotation_proof(box)
    minimum = np.tile(box.minimum, (instances.count(), 1))
    maximum = np.tile(box.maximum, (instances.count(), 1))
    if instances.scales is not None:
        scales = np.asarray(instances.scales, dtype='d')
        minimum, maximum = (np.minimum(minimum * scales, maximum * scales),
                            np.maximum(minimum * scales, maximum * scales))
    if instances.translations is not None:
        offsets = np.asarray(instances.translations, dtype='d')
        minimum, maximum = minimum + offsets, maximum + offsets
    return BoundingBox(minimum.min(axis=0), maximum.max(axis=0))
