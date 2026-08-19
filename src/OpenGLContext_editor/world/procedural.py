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

import math
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
from OpenGLContext.scenegraph.gantry import GantryProfile
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.props import Prop, rock_mesh
from OpenGLContext.scenegraph.road import RoadProfile
from OpenGLContext.scenegraph.roadsigns import SignProfile
from OpenGLContext.scenegraph.terrain import LayerRule

from OpenGLContext_editor.bake.assets import combined_mesh, meshes_from_gltf
from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.field import FieldTerrainLayer
from OpenGLContext_editor.bake.gantry import GantryLayer
from OpenGLContext_editor.bake.layers import (
    HeightfieldLayer,
    HeightFn,
    InstanceLayer,
    Layer,
    WaterLayer,
)
from OpenGLContext_editor.bake.props import PropLayer
from OpenGLContext_editor.bake.signs import SignLayer
from OpenGLContext_editor.bake.vegetation import VegetationLayer
from OpenGLContext_editor.world.gantry import StartFinish, start_finish
from OpenGLContext_editor.world.road import (
    RoadLayer,
    RoadPath,
    conform_terrain,
    conform_terrain_at,
    follow_terrain,
)
from OpenGLContext_editor.world.route import cornering_radius, ease_route
from OpenGLContext_editor.world.scatter import scatter_on_heightfield, yaw_quaternions
from OpenGLContext_editor.world.signs import sign_placements, warn_of
from OpenGLContext_editor.world.species import (
    biome_species,
    shipped_cover,
    shipped_credits,
    shipped_species,
)
from OpenGLContext_editor.world.structures import Op, choose_structures

#: Candidate trees per square metre, before thinning. They go one to a cell of a
#: jittered grid of ``1/sqrt(density)`` metres rather than at random, so the set
#: handed to the thinning is already nearly the answer.
#:
#: At random it would have to be several times denser before the thinning
#: saturated -- dart-throwing needs a lot of darts -- and every candidate costs
#: a height lookup, a distance-to-road query and four more lookups for the
#: slope. On a four-kilometre world that was eight million candidates for half a
#: million trees, and most of what a bake spent.
TREE_DENSITY = 0.148

#: How much room a tree keeps to itself, in metres: a fixed part plus a share of
#: its own height, so a mature tree holds more ground than a sapling. This is
#: what sets the density of the finished stand -- at these numbers a stand of
#: fifteen-metre firs closes up at about two metres between trunks, which is a
#: wood you cannot see through rather than trees standing on a lawn.
TREE_SPACING = 0.55
TREE_SPACING_PER_METRE = 0.085

#: Where trees will grow: above the waterline, below the treeline, and off
#: anything a tree would slide down. The treeline is quoted against the shipped
#: landscape's full relief and scales with it, so a gentler world keeps the same
#: proportion of bare tops rather than losing them. Trees hold ground steeper
#: than a road can climb, so the limit is what a root system gives up on, not
#: what a vehicle does.
TREE_ELEVATION = (WATER_LEVEL + 1.0, 330.0)
TREE_SLOPE_LIMIT = 52.0

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
CIRCUIT_SMOOTHING = 60.0
CIRCUIT_MAX_GRADE = 0.10

#: How far the drawn circuit may slide sideways to find ground a road can
#: follow, in metres, and how long it is given to find it. Without this the
#: ellipse crosses whatever the landscape puts in its way and most of the lap is
#: carried on structures; with it the line goes round the shoulder of a hill
#: instead of over it. See :mod:`OpenGLContext_editor.world.route`.
CIRCUIT_REACH = 300.0
CIRCUIT_EASING = 500

#: How far apart the alignment's points are, in metres. The plan is eased and
#: held to its corner radius at this spacing, because a plan of points tens of
#: metres apart is a polygon: the road along it turns through the whole of each
#: corner at one vertex, however gentle the polygon looks from a distance.
CIRCUIT_SPACING = 6.0

#: How fast the circuit is meant to be driven, in metres per second (151 km/h),
#: which is what rounds off its crests: a change of grade sharp enough to take a
#: car's wheels off the road at this speed is spread into a vertical curve that
#: does not.
CIRCUIT_DESIGN_SPEED = 42.0

#: How tall this world's hills are, as a multiple of the shipped landscape's own
#: relief. At 1 the terrain rises five hundred metres over four kilometres,
#: which no road held to a drivable grade can follow: a circuit across it is
#: viaduct and bore for most of its length. Halved, the same shapes make hill
#: country a road can be built through, with a handful of crossings where it
#: still cannot.
RELIEF = 0.5

#: The circuit stays this far above the waterline. Where the ground is lower --
#: the lake basin, the floor of the canyon -- the road rides over it on fill and
#: its approaches climb to meet it.
CAUSEWAY_FREEBOARD = 2.5

#: No tree stands closer to the road than its own half-width plus this, in
#: metres -- the cleared corridor a road is built inside. Narrow, because the
#: circuit is a forest road: the trees come up to the verge and the drive is
#: through them rather than past them.
ROAD_CLEARANCE = 0.8

#: Boulders per square metre, before anything is filtered out, and how big they
#: are. Sparse: a rock is a thing a driver notices, and a landscape strewn with
#: them evenly is a quarry rather than a hillside.
ROCK_DENSITY = 0.00035
ROCK_RADIUS = (0.45, 1.7)

#: How many different boulders are cut, and the tile error past which they are
#: not drawn. A handful is enough: they are turned, scaled and scattered, and a
#: driver seeing the same stone twice in a lap is not what anybody notices.
ROCK_SHAPES = 4

#: Where a boulder may lie: off the carriageway by its own size and this much
#: more, and no further out than this from the road. The near limit is what
#: makes it an obstacle rather than scenery -- something a car leaving the road
#: meets -- and the far one is what stops the whole landscape being strewn.
ROCK_CLEARANCE = 0.6
ROCK_REACH = 26.0

#: Boulders will not lie on ground steeper than this, in degrees, nor below the
#: waterline.
ROCK_SLOPE_LIMIT = 38.0

#: How far a mature crown reaches from its own trunk, in metres. Where the road
#: is on the land a crown over the carriageway is the point -- it is what closes
#: a forest road's canopy -- so the clearance is the corridor and no more. Where
#: the road is *carried*, a tree beside it is rooted metres below the surface and
#: the same reach goes through the structure instead of over the road, so it is
#: held back by this as well.
CROWN_RADIUS = 3.5

#: How far the road has to stand above the land under a tree before that tree is
#: growing into a structure rather than reaching over a road, in metres. Every
#: road stands a little proud of what it is built on.
CARRIED_ABOVE = 1.5

#: How much room is left round a gantry leg, in metres. A boulder inside the
#: upright holding the start line up is the one place on a circuit every driver
#: looks at.
GANTRY_CLEARANCE = 3.0

#: How far a sign's post stands outside the road's own edge, in metres. Inside
#: the cleared corridor, with room to spare: a forest road is cut only as wide as
#: it has to be, and a sign put outside that strip stands in the trees where
#: nobody sees it.
SIGN_OFFSET = ROAD_CLEARANCE * 0.6

#: The circuit's cross-section. Two lanes, a shoulder wide enough to put two
#: wheels on and no more, and a verge that is a strip rather than a field: a
#: road through a forest is cut only as wide as it has to be, and the ground
#: either side is the forest floor the trees stand on.
CIRCUIT_PROFILE = RoadProfile(lane_width=3.6, lanes=2,
                              shoulder_width=0.7, shoulder_drop=0.05,
                              verge_width=1.0, verge_drop=0.35,
                              texture_length=22.0)

#: How many pixels across the splat control map. Over four kilometres, 2048 is
#: two metres a pixel, which is what a twelve-metre road corridor needs to be a
#: corridor: at eight metres a pixel the carriageway is thinner than one pixel
#: and the ground cover grows straight over it.
CONTROL_SIZE = 2048

#: How many samples across the field terrain's height grid. 1025 over four
#: kilometres is four-metre spacing: fine enough that a road's cutting is a
#: cutting rather than a smoothed dip, and one mesh of about two million
#: triangles, which is a single draw and no shadow pass.
FIELD_RESOLUTION = 1025

#: What the ground is made of, and where each material belongs. Grass is the
#: fallback and covers the gentle ground; needle litter takes the middle slopes
#: where the forest is; rock takes anything too steep for soil to stay on; and
#: dirt is what the road's corridor is painted with, so the carriageway runs
#: through disturbed ground rather than out of a lawn.
GROUND_LAYERS = ('grass', 'forest_floor', 'rock', 'dirt')

#: Which of those the ground cover grows on: the two soft ones. Rock will not
#: hold it and the road's corridor is painted as dirt, so the grass stops at the
#: verge without being told where the road is.
COVER_ON = ('grass', 'forest_floor')
#: Where the shore is: dirt from the lake bed up to this far above the
#: waterline, and grass no lower. A band rather than a line, so the shore is a
#: beach that fades into the grass rather than an edge drawn round the water.
#:
#: **Nothing soft grows below it**, which is what keeps the ground cover out of
#: the lake: the cover grows on the two soft layers (:data:`COVER_ON`), so a
#: waterline the *layers* respect is one the grass respects without ever being
#: told where the water is.
SHORE_ABOVE = 1.5
SHORE_FEATHER = 2.5
GROUND_RULES = (
    LayerRule(height=(WATER_LEVEL + SHORE_ABOVE, 1.0e9), feather=SHORE_FEATHER),
    LayerRule(height=(WATER_LEVEL + SHORE_ABOVE, 1.0e9),
              slope=(0.16, 0.55), weight=1.5, feather=SHORE_FEATHER),
    LayerRule(slope=(0.5, 1.0e9), weight=3.0),
    LayerRule(height=(-1.0e9, WATER_LEVEL + SHORE_ABOVE), weight=4.0,
              feather=SHORE_FEATHER),
)

CREDITS = (
    "Terrain and road surface: generated procedurally by OpenGLContext "
    "(BSD-3-Clause).",
)

#: What each way of carrying the forest has to say for itself.
TILE_TREE_CREDITS = (
    "Tree geometry and foliage textures: generated procedurally by "
    "OpenGLContext (BSD-3-Clause).",
)
FIELD_GROUND_CREDITS = (
    "Ground detail materials: ambientCG (CC0 1.0), fetched and cached at first "
    "use.",
)


@dataclass
class ProceduralWorld:
    """The knobs on the shipped example world.

    ``extent`` is the side of the square it covers, in metres, centred on the
    origin. ``resolution`` is how many ground samples across each tile gets --
    the vertex budget that, divided by the tile's size, sets the detail.

    ``route`` is the circuit's plan as an (N,2) array of XZ points -- what a
    designer drew. Left out, the world draws its own, and slides it onto ground
    a road can follow (``ease``). Either way it is a *plan*: it arrives with no
    heights on it, and everything else about assembling the world is the same,
    which is the point of it being one argument rather than a second class.

    ``structures`` decides whether the alignment's large departures from the
    land are built as bridges and tunnels. With it off the same road is carried
    entirely on earthworks, which over a landscape of this relief means
    embankments and cuttings the size of the hills they cross.

    ``forest`` is how the trees are carried. ``'field'`` writes them as one
    table beside the tileset, drawn by the runtime as real geometry near the
    camera and cards beyond it -- which is how a forest of a quarter of a
    million costs what it does. ``'tiles'`` places them into the tile tree
    instead, as instanced copies of one procedural conifer, which needs no tree
    assets at all.

    ``ground`` is how the landscape is carried. ``'field'`` writes it once as a
    height image and a splat control map beside the tileset, which a viewer
    draws as a single splat terrain: crisp detail materials up to the camera,
    one draw call, and no ground in any tile. ``'tiles'`` meshes it into the
    tile tree instead, vertex-coloured, which is what a world too large to hold
    at once needs.
    """

    extent: float = 4096.0
    resolution: int = 33
    tree_density: float = TREE_DENSITY
    tree_height: float = 15.0
    seed: int = 11
    road: bool = True
    #: The circuit's plan, (N,2) XZ; None for the world's own.
    route: Any = None
    #: Whether the route returns to where it started.
    closed: bool = True
    #: Whether a deck or a bore is built where the earthworks would be huge.
    structures: bool = True
    #: How the ground is carried: 'field' (one splat terrain) or 'tiles'.
    ground: str = 'field'
    #: How many samples across the field's height grid.
    field_resolution: int = FIELD_RESOLUTION
    #: How many pixels across the splat control map.
    control_size: int = CONTROL_SIZE
    #: Where the water sits, in metres. Ground below it is a lake bed with a
    #: sheet of water over it; the circuit is held clear of it by
    #: :data:`CAUSEWAY_FREEBOARD`. Raise it to flood the valleys.
    water_level: float = WATER_LEVEL
    #: How tall the hills are, against the shipped landscape's own relief.
    relief: float = RELIEF
    #: Whether the world's *own* circuit is slid onto ground a road can follow.
    #: A ``route`` a caller gives is built as it was drawn either way: it is a
    #: designer's line, and moving it is the designer's decision to make.
    ease: bool = True
    #: How the trees are carried: 'field' (a table and its species) or 'tiles'.
    forest: str = 'field'
    #: Where the species' files are; None for the shipped ones.
    species_directory: str | None = None
    wetness: float = 0.0
    _circuit: RoadPath | None = field(default=None, init=False, repr=False)
    _terrain: Layer | None = field(default=None, init=False, repr=False)
    _scatter: Any = field(default=None, init=False, repr=False)
    _start_line: StartFinish | None = field(default=None, init=False,
                                            repr=False)

    def natural(self) -> HeightFn:
        """The land before the road touched it, at this world's relief."""
        if self.relief == 1.0:
            shipped: HeightFn = terrain_height
            return shipped

        def scaled(x: Any, z: Any) -> Any:
            return np.asarray(terrain_height(x, z), dtype='d') * self.relief
        return scaled

    def treeline(self) -> tuple[float, float]:
        """The band trees grow in, scaled with the world's own relief."""
        low, high = TREE_ELEVATION
        return (low, high * self.relief)

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
        layers: list[Layer] = [self.terrain(), self.water(), self.trees()]
        if self.road:
            layers.append(self.circuit_layer())
            signs = self.sign_layer()
            if signs is not None:
                layers.append(signs)
            layers.append(self.gantry_layer())
        props = self.prop_layer()
        if props is not None:
            layers.append(props)
        return layers

    def water(self) -> Layer:
        """The lakes: a sheet wherever the ground dips below the waterline.

        Its own surface rather than the ground clamped flat, which is what
        gives the world a shoreline -- the shore is where the land passes
        through the water, and there is no such line on a ground that has been
        levelled at it.
        """
        return WaterLayer(height_fn=self.height_fn(), extent=self.footprint(),
                          level=self.water_level, name='water')

    def rocks(self) -> Any:
        """Where the boulders lie: near the road, clear of the carriageway.

        Near it because that is what makes a rock an obstacle rather than
        scenery: it is the thing a car leaving the road meets. Clear of the
        carriageway because an obstacle a driver cannot avoid is not an
        obstacle, it is a wall.
        """
        from OpenGLContext.loaders.tiles3d.scatter import Scatter
        placed = scatter_on_heightfield(
            self.height_fn(), self.footprint(), density=ROCK_DENSITY,
            seed=self.seed + 101, scale_range=ROCK_RADIUS,
            slope_limit=ROCK_SLOPE_LIMIT, slope_fn=self.slope_fn(),
            height_range=(WATER_LEVEL + 0.5, 1.0e9),
            keep=self._beside_the_road)
        return Scatter(placed.positions, placed.yaws, placed.scales)

    def prop_layer(self) -> PropLayer | None:
        """The world's obstacles, or None for a world with nothing in the way."""
        placed = self.rocks()
        if not len(placed.positions):
            return None
        prototypes = {_rock_kind(index): rock_mesh(radius=1.0, seed=index)
                      for index in range(ROCK_SHAPES)}
        props = [
            Prop.of(prototypes[_rock_kind(index % ROCK_SHAPES)],
                    kind=_rock_kind(index % ROCK_SHAPES),
                    position=point, yaw=float(yaw), scale=float(scale))
            for index, (point, yaw, scale) in enumerate(
                zip(placed.positions, placed.yaws, placed.scales, strict=True))]
        return PropLayer(props=props, prototypes=prototypes)

    def _beside_the_road(self, points: np.ndarray) -> np.ndarray:
        """Which placements are near the road but out of the way of a car.

        Without a road every one of them stands: a landscape has rocks in it
        whether or not anybody built through it.
        """
        if not self.road:
            return np.ones(len(points), dtype=bool)
        circuit = self.circuit()
        clear = circuit.profile.carriageway_width / 2.0 + ROCK_RADIUS[1] \
            + ROCK_CLEARANCE
        found = circuit.sample(points[:, 0], points[:, 2],
                               radius=ROCK_REACH * 1.5)
        keep = np.asarray((found.distance > clear)
                          & (found.distance < ROCK_REACH))
        room: np.ndarray = keep & self._clear_of_the_gantry(points)
        return room

    def _clear_of_the_gantry(self, points: np.ndarray) -> np.ndarray:
        """Which placements leave the start line's uprights room to stand."""
        line = self.start_line()
        beam = np.array([math.cos(line.yaw), 0.0, -math.sin(line.yaw)])
        at = np.asarray(line.position, dtype='d')
        room = np.ones(len(points), dtype=bool)
        for side in (-1.0, 1.0):
            foot = at + beam * (side * line.span / 2.0)
            away = points[:, [0, 2]] - foot[[0, 2]]
            room &= np.einsum('ij,ij->i', away, away) > GANTRY_CLEARANCE ** 2
        return room

    def sign_layer(self) -> SignLayer | None:
        """The circuit's warning signs, or None for a world with no road.

        Nothing here decides what they say: the alignment does, from its own
        curvature, its own grade and the structures along it. See
        :mod:`OpenGLContext_editor.world.signs`.
        """
        if not self.road:
            return None
        circuit = self.circuit()
        warnings = warn_of(circuit, CIRCUIT_DESIGN_SPEED)
        if not warnings:
            return None
        profile = SignProfile(offset=SIGN_OFFSET)
        return SignLayer(sign_placements(circuit, warnings, profile=profile,
                                         ground=self.height_fn()),
                         profile=profile)

    def start_line(self) -> StartFinish:
        """Where the circuit's start/finish gantry stands.

        Where the centreline begins, which for a closed circuit is where a lap
        begins and ends. A road that does not return to its start is marked at
        the point it sets off from.
        """
        if self._start_line is None:
            self._start_line = start_finish(self.circuit(),
                                            profile=GantryProfile(),
                                            ground=self.height_fn())
        return self._start_line

    def gantry_layer(self) -> GantryLayer:
        """The circuit's start/finish marker: a beam over the road and a line
        painted under it."""
        return GantryLayer(placement=self.start_line())

    def height_fn(self) -> Any:
        """The ground as the world finally has it, earthworks included."""
        if not self.road:
            return self.natural()
        return conform_terrain(self.natural(), self.circuit())

    def circuit(self) -> RoadPath:
        """The race circuit: laid out on the natural ground, smoothed, and told
        which of its stretches are carried rather than laid."""
        if self._circuit is None:
            ground = self.natural()
            plan = (np.asarray(self.route, dtype='d') if self.route is not None
                    else circuit_plan(self.extent * 0.32, self.extent * 0.25))
            if self.ease and self.route is None:
                plan = ease_route(
                    plan, ground, reach=CIRCUIT_REACH, rounds=CIRCUIT_EASING,
                    closed=self.closed, spacing=CIRCUIT_SPACING,
                    minimum_radius=cornering_radius(CIRCUIT_DESIGN_SPEED))
            line = follow_terrain(plan, ground, spacing=CIRCUIT_SPACING,
                                  smoothing=CIRCUIT_SMOOTHING,
                                  maximum_grade=CIRCUIT_MAX_GRADE,
                                  design_speed=CIRCUIT_DESIGN_SPEED,
                                  minimum_height=WATER_LEVEL + CAUSEWAY_FREEBOARD,
                                  closed=self.closed)
            self._circuit = RoadPath(line, profile=CIRCUIT_PROFILE,
                                     ops=self._ops(line))
        return self._circuit

    def _ops(self, line: np.ndarray) -> Any:
        """What is built along the alignment, point by point."""
        if not self.structures:
            return None
        natural = np.asarray(self.natural()(line[:, 0], line[:, 2]), dtype='d')
        chosen = choose_structures(line, natural, waterline=WATER_LEVEL,
                                   closed=self.closed)
        ops = np.full(len(line), Op.DIRT, dtype=object)
        for structure in chosen:
            ops[structure.indices(len(line))] = structure.kind
        return ops

    def circuit_layer(self) -> RoadLayer:
        return RoadLayer(self.circuit(), wetness=self.wetness,
                         ground=self.natural(), shade=self.canopy_shade())

    def terrain(self) -> Layer:
        """The ground, as whichever kind of terrain layer the world asked for."""
        if self._terrain is None:
            self._terrain = self._build_terrain()
        return self._terrain

    def _build_terrain(self) -> Layer:
        if self.ground == 'field':
            return self.field_terrain()
        if self.ground != 'tiles':
            raise ValueError("a world's ground is 'field' or 'tiles', not %r"
                             % (self.ground,))
        return HeightfieldLayer(
            height_fn=self.height_fn(), height_fn_at=self.height_fn_at(),
            extent=self.footprint(), resolution=self.resolution,
            # No clamp: the ground is meshed as it is and the water is laid
            # over it, so the shoreline is where the land actually passes
            # through the surface.
            color_fn=terrain_colors, water_level=None, name='terrain')

    def _tree_slopes(self, positions: Any) -> Any:
        """How steep the ground is under each tree, as rise over run."""
        steepness = self.slope_fn()
        if steepness is None:
            return _slopes(self.height_fn(), positions)
        return np.tan(np.asarray(steepness(positions[:, 0], positions[:, 2]),
                                 dtype='d'))

    def slope_fn(self) -> Any:
        """How steep the ground is, in radians, or None for a world with no field.

        The landscape is sampled into a height field before anything is
        scattered on it, and asking that is a lookup rather than four
        evaluations of the conformed height function. At four metres a sample it
        also answers the question a tree asks -- whether the hillside is too
        steep to hold one -- rather than whether the metre it stands on is.
        """
        layer = self.terrain()
        if not isinstance(layer, FieldTerrainLayer):
            return None
        field = layer.field()

        def steepness(x: Any, z: Any) -> Any:
            return np.arctan(np.asarray(field.slope(x, z), dtype='d'))
        return steepness

    def canopy_shade(self) -> Any:
        """How much of the sun reaches each place: ``shade(x, z)`` in [0, 1].

        The hillside's own shadow with the forest's over it, worked out here so
        that what a bake writes into the road agrees with what the viewer draws
        the ground and the plants with -- it is the same node answering, with
        the same trunks and the same sun. None where the world has no landscape
        or no forest to shade it with.
        """
        from OpenGLContext.scenegraph.terrain.splat import SplatTerrain
        layer = self.terrain()
        if not isinstance(layer, FieldTerrainLayer) or self.forest != 'field':
            return None
        return SplatTerrain(layer.field(), list(GROUND_LAYERS), control=None,
                            canopy=self.scatter().positions).shade

    def field_terrain(self) -> FieldTerrainLayer:
        """The landscape as one height field and one splat control map."""
        return FieldTerrainLayer(
            height_fn=self.height_fn(), height_fn_at=self.height_fn_at(),
            extent=self.footprint(), resolution=self.field_resolution,
            control_size=self.control_size,
            layers=list(GROUND_LAYERS), rules=list(GROUND_RULES),
            road=self.circuit() if self.road else None,
            road_layer=GROUND_LAYERS.index('dirt'),
            road_corridor=self._corridor(), name='terrain')

    def _corridor(self) -> float:
        """How far out the road's own ground reaches, in metres.

        Out to where the trees start, so the bare strip beside the carriageway
        is the strip that was cleared for it and the forest floor begins where
        the forest does.
        """
        across: float = CIRCUIT_PROFILE.total_width
        return across / 2.0 + ROAD_CLEARANCE

    def height_fn_at(self) -> Any:
        """The ground as a function of the spacing a tile samples it at.

        A cutting narrower than a coarse tile's vertex spacing would be stepped
        over and the road in it buried, so each tile gets the earthwork widened
        to its own resolution.
        """
        if not self.road:
            return None
        return conform_terrain_at(self.natural(), self.circuit())

    def scatter(self) -> Any:
        """Where the trees stand: on the finished ground, clear of the road.

        Thinned to blue noise afterwards, so no two trees are closer than the
        room their heights ask for and the biggest in a crowd keeps the spot.
        """
        if self._scatter is None:
            self._scatter = self._build_scatter()
        return self._scatter

    def _build_scatter(self) -> Any:
        from OpenGLContext.loaders.tiles3d.scatter import Scatter
        from OpenGLContext.loaders.tiles3d.vegetation import poisson_thin
        if self.tree_density <= 0.0:
            return Scatter(np.zeros((0, 3), 'f4'), np.zeros(0), np.zeros(0))
        placed = scatter_on_heightfield(
            self.height_fn(), self.footprint(),
            spacing=1.0 / math.sqrt(self.tree_density),
            seed=self.seed, scale_range=(0.62, 1.30),
            slope_limit=TREE_SLOPE_LIMIT, height_range=self.treeline(),
            slope_fn=self.slope_fn(), keep=self._away_from_the_road)
        if not len(placed.positions):
            return placed
        heights = placed.scales * self.tree_height
        keep = poisson_thin(placed.positions,
                            TREE_SPACING + TREE_SPACING_PER_METRE * heights,
                            priority=heights)
        return Scatter(placed.positions[keep], placed.yaws[keep],
                       placed.scales[keep])

    def trees(self) -> Layer:
        """The forest, as whichever kind of vegetation layer was asked for."""
        if self.forest == 'field':
            return self.tree_field()
        if self.forest != 'tiles':
            raise ValueError("a world's forest is 'field' or 'tiles', not %r"
                             % (self.forest,))
        placed = self.scatter()
        return InstanceLayer(
            positions=placed.positions,
            rotations=yaw_quaternions(placed.yaws),
            scales=placed.scales,
            lods=[(0.0, conifer_mesh(self.tree_height, self.seed)),
                  (IMPOSTOR_ERROR, impostor_mesh(self.tree_height, self.seed))],
            name='trees')

    def tree_field(self) -> VegetationLayer:
        """The forest as a table beside the tileset, drawn by species.

        Which kind of tree stands where is decided from the ground under it, so
        conifers take the high and the steep and the broadleaves the valleys.
        """
        placed = self.scatter()
        species = shipped_species(self.species_directory)
        heights = (placed.scales * self.tree_height).astype('f4')
        slopes = self._tree_slopes(placed.positions)
        return VegetationLayer(
            positions=placed.positions, yaws=placed.yaws, heights=heights,
            species=species,
            species_id=biome_species(placed.positions, heights, slopes,
                                     seed=self.seed),
            cover=self.ground_cover(), cover_on=list(COVER_ON),
            name='trees')

    def ground_cover(self) -> Any:
        """What grows on the ground between the trees, or None for bare ground.

        A recipe rather than a scatter: see
        :class:`~OpenGLContext.scenegraph.vegetation.cover.GroundCover`. It comes
        from the same place the species do, so a world pointed at its own art
        gets its own grass.
        """
        try:
            return shipped_cover(self.species_directory)
        except LookupError:
            return None

    def credits(self) -> list[str]:
        """Where everything in this world came from, licences included.

        What a world has to say depends on what it is made of, so the list is
        assembled from the choices rather than written down once: a world with
        no CC-BY trees in it should not claim any, and one with them must.
        """
        found = list(CREDITS)
        if self.ground == 'field':
            found.extend(FIELD_GROUND_CREDITS)
        found.extend(shipped_credits() if self.forest == 'field'
                     else TILE_TREE_CREDITS)
        return found

    def _away_from_the_road(self, points: np.ndarray) -> np.ndarray:
        """Which placements are outside the road's cleared corridor.

        Wider where the road is carried above the land the tree stands on: the
        crown of a tree rooted at the foot of an embankment grows through the
        side of it rather than over the carriageway, and a wood growing out of
        a causeway's concrete is what that looks like from the road.
        """
        if not self.road:
            return np.ones(len(points), dtype=bool)
        circuit = self.circuit()
        corridor = circuit.profile.total_width / 2.0 + ROAD_CLEARANCE
        reach = corridor + CROWN_RADIUS
        found = circuit.sample(points[:, 0], points[:, 2], radius=reach * 1.5)
        ground = np.asarray(self.natural()(points[:, 0], points[:, 2]),
                            dtype='d')
        carried = found.height - ground > CARRIED_ABOVE
        return np.asarray(found.distance > np.where(carried, reach, corridor))


def _rock_kind(index: int) -> str:
    """The name of one of the cut boulders, as a prop's kind."""
    return 'rock%d' % (index,)


def _slopes(height_fn: Any, positions: np.ndarray, step: float = 8.0
            ) -> np.ndarray:
    """How steep the ground is under each placement, as rise over run."""
    x, z = positions[:, 0], positions[:, 2]
    dx = (np.asarray(height_fn(x + step, z), dtype='d')
          - np.asarray(height_fn(x - step, z), dtype='d')) / (2.0 * step)
    dz = (np.asarray(height_fn(x, z + step), dtype='d')
          - np.asarray(height_fn(x, z - step), dtype='d')) / (2.0 * step)
    return np.hypot(dx, dz)


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
