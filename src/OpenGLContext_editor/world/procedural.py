"""A world you can bake without authoring anything first.

The layers here describe a forested landscape -- hills, ridges, a river canyon
and a lake basin, with conifers on the ground that will hold them -- assembled
entirely from what the engine already generates. It is the worked example the
toolkit ships: ``glisteel-bake`` bakes it, ``oglc-view`` streams the result, and the
code below is the shortest honest answer to "how do I describe a world?".

Every piece of it is a normal layer, so a world of your own is this file with
your height function, your assets and your scatter rules.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from OpenGLContext.loaders.tiles3d import foliage
from OpenGLContext.loaders.tiles3d.procedural import (
    WATER_LEVEL,
    terrain_colors,
)
from OpenGLContext.scenegraph.gantry import GantryProfile
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.props import Prop, RockProfile, rock_mesh
from OpenGLContext.scenegraph.road import (
    MAXIMUM_BANK,
    RoadProfile,
    bank_profile,
)
from OpenGLContext.scenegraph.roadsigns import SignProfile
from OpenGLContext.scenegraph.roadworks import (
    BORE_APPROACH_CELLS,
    TunnelProfile,
)
from OpenGLContext.scenegraph.terrain import GROUND_RELIEF, LayerRule, Relief

from OpenGLContext_editor.bake.assets import combined_mesh, meshes_from_gltf
from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.driver import BakeResult, bake_world, root_error_for
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
from OpenGLContext_editor.bake.zones import ZonesLayer, place_sounds, zone_records
from OpenGLContext_editor.world.character import corner_radii, road_character
from OpenGLContext_editor.world.gantry import StartFinish, start_finish
from OpenGLContext_editor.world.height import (
    DEFAULT_RELIEF,
    HeightSource,
    ProceduralBase,
)
from OpenGLContext_editor.world.places import road_places
from OpenGLContext_editor.world.road import (
    RoadLayer,
    RoadPath,
    conform_terrain,
    conform_terrain_at,
    follow_terrain,
)
from OpenGLContext_editor.world.route import (
    cornering_radius,
    ease_route,
    hold_corners,
)
from OpenGLContext_editor.world.scatter import scatter_on_heightfield, yaw_quaternions
from OpenGLContext_editor.world.signs import (
    POSTED_LIMIT,
    sign_placements,
    warn_of,
)
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

#: Where trees will grow: this far above the world's own waterline, below the
#: treeline, and off anything a tree would slide down. The treeline is quoted
#: against the shipped landscape's full relief and scales with it, so a gentler
#: world keeps the same proportion of bare tops rather than losing them. Trees
#: hold ground steeper than a road can climb, so the limit is what a root system
#: gives up on, not what a vehicle does.
TREE_FREEBOARD = 1.0
TREELINE = 330.0
TREE_SLOPE_LIMIT = 52.0

#: The tile error at which a conifer becomes two crossed cards. Roughly the
#: point at which a tree covers a few pixels, so the swap is not seen.
IMPOSTOR_ERROR = 8.0

#: The circuit: an oval bent out of round by two harmonics, so it has long
#: straights, a pair of fast sweepers and one slower complex, rather than being
#: a ring the car can hold at full throttle.
CIRCUIT_HARMONICS = ((3, 0.20), (5, 0.09))

#: How many corners the circuit has, and so how many straights join them.
#: Enough that a lap has variety in it; few enough that the legs between them
#: are long -- a straight is what a circuit is overtaken on, and one shorter
#: than the manoeuvre is a straight nobody passes on.
CIRCUIT_CORNERS = 7

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

#: How fast the circuit is meant to be driven, in metres per second: two hundred
#: kilometres an hour.
#:
#: It decides two things about the shape of the road. Its corners are held to
#: the radius this speed needs (:func:`~OpenGLContext.scenegraph.road.cornering_radius`,
#: 315 m), and its crests are rounded off so that a change of grade sharp enough
#: to take a car's wheels off the road at this speed is spread into a vertical
#: curve that does not.
#:
#: It is the speed the circuit is *for*: a road whose corners are worth a
#: hundred and fifty is a road nobody averages two hundred on, however hard they
#: drive it.
CIRCUIT_DESIGN_SPEED = 200.0 / 3.6

#: How far in or out a varied circuit's vertices are drawn from the ellipse
#: they sit on, as a fraction of its radius, at full ``variation``. A vertex
#: inside its neighbours turns the road much further than one in line with
#: them; a fifth is enough for a near-reversal without folding the loop back
#: through itself.
CIRCUIT_EXCURSION = 0.20

#: The shortest step a varied circuit puts between two vertices, as a share of
#: an even one. Two corners a fifth of an even step apart are a chicane; two on
#: top of each other are one corner drawn twice.
CIRCUIT_LEAST_STEP = 0.2

#: How unlike each other the circuit's corners and stretches are, from 0 to 1.
#:
#: At 0 the circuit is a regular loop of one corner repeated, laid out to one
#: design speed, one grade and one smoothing from end to end -- the road every
#: figure being a single figure gives. Turned up, the vertices are drawn apart
#: (:func:`circuit_plan`), the corners are drawn from a mix
#: (:data:`~OpenGLContext_editor.world.character.CORNER_MIX`) so a lap gets a
#: hairpin and a sweeper as well as the corner it was laid out for, and each
#: stretch is then laid out for the speed *it* is worth -- which is what puts
#: bumps on the slow parts, a climb where the land climbs, and a clearing at
#: the corners a driver has to see round.
CIRCUIT_VARIETY = 1.0

#: The steepest the circuit may climb where the land climbs harder than
#: :data:`CIRCUIT_MAX_GRADE`, as a fraction. One in seven is a hill road: a car
#: gets up it and a driver knows they are on it. Held to the ordinary limit
#: everywhere the land does not demand more, so this is what a hillside buys
#: rather than what the road is.
CIRCUIT_STEEP_GRADE = 0.14

#: How far the circuit's corners lean, at most, as a fraction -- how far the
#: road's surface rises across it over the distance it rises across.
#:
#: :data:`~OpenGLContext.scenegraph.road.MAXIMUM_BANK`, which is the steepest a
#: **road** is built: this is a road through hill country driven fast, not an
#: oval. Flat, a corner holding :data:`CIRCUIT_DESIGN_SPEED` needs a radius of
#: 315 m; leaning this far it needs 257 m, and the corners the landscape
#: already had hold some ten per cent more speed than they did.
#:
#: What each corner actually gets is the lean that *balances* a car at the
#: design speed and no more (:func:`~OpenGLContext.scenegraph.road.bank_profile`),
#: capped here: a gentle sweeper leans a little, a tight corner leans to the
#: limit, and no corner is banked for a speed nothing on it will do. Zero lays
#: the circuit out flat, which is the road the design speed alone would give.
CIRCUIT_MAXIMUM_BANK = MAXIMUM_BANK

#: How much looser than the theoretical minimum the circuit's corners are laid
#: out. The design speed is a **floor**, and a corner at exactly the radius that
#: floor asks for holds it with nothing to spare -- so a corner built a fraction
#: tighter than it was drawn holds a fraction less. A plan is drawn at one
#: spacing, filleted, draped over the ground and re-sampled, and each of those
#: moves the line by centimetres; a twentieth is more room than all of them
#: together need, and costs a corner nothing anybody driving it would notice.
CORNER_MARGIN = 1.05

#: How tall this world's hills are, as a multiple of the shipped landscape's own
#: relief. At 1 the terrain rises five hundred metres over four kilometres,
#: which no road held to a drivable grade can follow: a circuit across it is
#: viaduct and bore for most of its length. Halved, the same shapes make hill
#: country a road can be built through, with a handful of crossings where it
#: still cannot.
RELIEF = DEFAULT_RELIEF

#: The circuit stays this far above the waterline. Where the ground is lower --
#: the lake basin, the floor of the canyon -- the road rides over it on fill and
#: its approaches climb to meet it.
CAUSEWAY_FREEBOARD = 2.5

#: No tree stands closer to the road than its own half-width plus this, in
#: metres -- the cleared corridor a road is built inside. Narrow, because the
#: circuit is a forest road: the trees come up to the verge and the drive is
#: through them rather than past them.
ROAD_CLEARANCE = 0.8

#: How much further out than the road's own corridor the trees may be cut back
#: where a driver needs to see round a bend, in metres. Enough to open a corner
#: out; not so much that the drive stops being through a forest.
MOST_CLEARING_BEYOND = 6.0

#: How much more carriageway a sustained climb gets, in metres -- a lane's
#: worth, so there is room to get past whatever is labouring up it. The corridor
#: opens out with it, since a wider road inside the same cleared strip is a road
#: whose new lane runs into the trees.
CLIMBING_LANE = 3.6

#: Boulders per square metre, before anything is filtered out, and how big they
#: are. Sparse: a rock is a thing a driver notices, and a landscape strewn with
#: them evenly is a quarry rather than a hillside.
ROCK_DENSITY = 0.00035
ROCK_RADIUS = (0.45, 1.7)

#: How many different boulders are cut, and the tile error past which they are
#: not drawn. A handful is enough: they are turned, scaled and scattered, and a
#: driver seeing the same stone twice in a lap is not what anybody notices.
ROCK_SHAPES = 4

#: Loose stone per square metre, how big one is in metres, and how many shapes
#: are cut. This is what a hillside is made of rather than what a car runs into:
#: knee-high at most, meshed into the tiles that are close enough to draw it
#: (:class:`~OpenGLContext_editor.bake.stones.StoneLayer`), and stood up by a
#: game as a dome a wheel rides over rather than a block that stops it.
#:
#: Dense enough that a hillside seen from a standing start has stone in it, and
#: sparse enough that the world is a landscape and not a scree slope: about one
#: every sixteen metres square.
STONE_DENSITY = 0.004
STONE_RADIUS = (0.10, 0.35)
STONE_SHAPES = 5

#: How rough a hillside a stone lies on, in degrees, and how much room it
#: keeps outside the cleared corridor, in metres. Steeper than that and it
#: would have rolled; inside the clearing it would be stone a machine took
#: away when the road was built.
STONE_SLOPE_LIMIT = 46.0
STONE_CLEARANCE = 1.5

#: Over how many metres the ground's own grain comes back beyond the road's
#: edge. A grader leaves a batter and not a cliff, and grain that switches on
#: across one cell is a ridge down the length of the road.
GRAIN_FADE = 12.0

#: How far the ground has to stand clear of the road, in metres, before it
#: counts as ground the road merely passes rather than ground it was built on.
#: A bore's hill and a deck's valley are tens of metres clear; a portal and an
#: abutment are not, and those are where a hummock stands in the carriageway.
#: Ground within :data:`GRAIN_AT_GRADE` of the road is all of it graded, which
#: is what makes the carriageway exactly smooth rather than nearly so -- the
#: formation a road is built on sits a little under its surface, and a fade
#: measured from the surface would leave a trace of grain on the road.
GRAIN_ABOVE = 8.0
GRAIN_AT_GRADE = 2.0

#: How far a stone is bedded into the ground, as a share of its own size.
#: Stone lying on a hillside is settled into it; one resting on the surface at
#: a single point reads as scenery dropped from above. What is drawn and what
#: is stood on are both measured from the bedded foot, so this moves the two
#: together.
STONE_SETTLE = 0.2

#: What a loose stone is cut as. No subdivision at all, against a boulder's
#: two: a stone is drawn flat-faceted, so every triangle carries its own three
#: vertices, and at knee height twenty faces is a stone while eighty is eighty
#: faces nobody counts. A tile carrying fifty of them pays the difference
#: fifty times.
STONE_PROFILE = RockProfile(facets=0, roughness=0.4, settled=0.4)

#: Where a boulder may lie: outside the strip cleared for the road by its own
#: size and this much more, and no further out than this from it. The near
#: limit is what makes it an obstacle rather than scenery -- something a car
#: leaving the road meets -- and the far one is what stops the whole landscape
#: being strewn.
ROCK_CLEARANCE = 0.6
ROCK_REACH = 26.0

#: Boulders will not lie on ground steeper than this, in degrees, nor nearer
#: the world's own waterline than this, in metres.
ROCK_SLOPE_LIMIT = 38.0
ROCK_FREEBOARD = 0.5

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


def ground_rules(water_level: float = WATER_LEVEL) -> tuple[LayerRule, ...]:
    """Where each ground material belongs, for a world with its water there.

    Taken from the world rather than written down once, because every one of
    these rules is quoted against the shore: a world that floods its valleys
    and paints the shore at sea level has a beach forty metres under water and
    grass growing over the whole of the sea bed.
    """
    shore = float(water_level) + SHORE_ABOVE
    return (
        LayerRule(height=(shore, 1.0e9), feather=SHORE_FEATHER),
        LayerRule(height=(shore, 1.0e9),
                  slope=(0.16, 0.55), weight=1.5, feather=SHORE_FEATHER),
        LayerRule(slope=(0.5, 1.0e9), weight=3.0),
        LayerRule(height=(-1.0e9, shore), weight=4.0, feather=SHORE_FEATHER),
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
    #: What the circuit is posted at, in km/h, repeated along it on speed limit
    #: signs. How fast a *bend* is worth the road works out for itself; what the
    #: whole road is posted at is a decision, so it is told. 0 leaves the road
    #: unposted.
    posted: int = POSTED_LIMIT
    #: The rivers running through this world, as
    #: :class:`~OpenGLContext_editor.world.hydrology.Channel` beds. Their water
    #: is a layer; their *bed* is already in the height source, because a
    #: channel is an edit on it.
    channels: Any = field(default_factory=list)
    #: The circuit's plan, (N,2) XZ; None for the world's own.
    route: Any = None
    #: Whether the route returns to where it started.
    closed: bool = True
    #: The world point a lap begins at, as ``(x, z)``. None puts the line where
    #: the centreline starts, which for a drawn circuit is its first point.
    start_at: tuple[float, float] | None = None
    #: Whether a deck or a bore is built where the earthworks would be huge.
    structures: bool = True
    #: How the ground is carried: 'field' (one splat terrain) or 'tiles'.
    ground: str = 'field'
    #: The grain in tiled ground: the hummocks and ruts a landscape has that a
    #: height function a kilometre wide does not carry
    #: (:class:`~OpenGLContext.scenegraph.terrain.Relief`). Each tile is meshed
    #: with as much of it as its own samples can show and no more than its
    #: geometric error allows, so refining the tree buys detail rather than a
    #: denser sampling of the same smooth surface. None leaves the tiles smooth.
    #: The landscape beside the tileset -- what a car is driven on and a tree is
    #: planted on -- carries the bands its own grid can hold, at the finest
    #: tile's figures, which is what the finest tile draws (:meth:`grain_drawn`).
    grain: Relief | None = GROUND_RELIEF
    #: How many times the tile tree subdivides when this world is baked. It is
    #: the bake's setting, and the world is told it because a tiled world's
    #: ground is only as fine as its deepest tile: how wide a portal's face has
    #: to be, and how far in front of one the ground is cleared, are measured
    #: against a cell of the finest ground the world will carry
    #: (:meth:`ground_spacing`).
    depth: int = 4
    #: How many samples across the field's height grid.
    field_resolution: int = FIELD_RESOLUTION
    #: How many pixels across the splat control map.
    control_size: int = CONTROL_SIZE
    #: Where the water sits, in metres. Ground below it is a lake bed with a
    #: sheet of water over it. Raise it to flood the valleys, and the whole
    #: world moves with it: the circuit is held :data:`CAUSEWAY_FREEBOARD`
    #: above it and the fill it crosses the drowned ground on is built as a
    #: causeway, the forest starts at :meth:`treeline`, the boulders lie no
    #: lower, and the shore is painted round the new coast
    #: (:func:`ground_rules`), which is what keeps the ground cover out of the
    #: sea.
    water_level: float = WATER_LEVEL
    #: How far the circuit's corners lean, at most, as a fraction
    #: (:data:`CIRCUIT_MAXIMUM_BANK`). Zero lays it out flat, which gives the
    #: long sweeping corners the design speed needs with no help from the road.
    maximum_bank: float = CIRCUIT_MAXIMUM_BANK
    #: How unlike each other the circuit's corners and stretches are, from 0 to
    #: 1 (:data:`CIRCUIT_VARIETY`). Zero is the road one figure for everything
    #: gives.
    variety: float = CIRCUIT_VARIETY
    #: How tall the hills are, against the shipped landscape's own relief.
    #: Ignored when a ``source`` is given, which carries its own.
    relief: float = RELIEF
    #: Where the ground comes from: a base and the edits made to it. None for
    #: the shipped landscape at this world's ``relief``, which is what a world
    #: nobody has authored has.
    source: HeightSource | None = None
    #: Whether the world's *own* circuit is slid onto ground a road can
    #: follow. A ``route`` a caller gives is built as it was drawn either way:
    #: it is a designer's line, and moving it is the designer's decision to
    #: make.
    #:
    #: Off, because the world's own circuit is drawn too now -- straights
    #: joined by corners of the radius :data:`CIRCUIT_DESIGN_SPEED` asks for
    #: (:func:`circuit_plan`) -- and sliding is for a route that was *found*,
    #: where the shape is an artefact of the search. Slid, the straights come
    #: back as gentle curves and the overtaking goes with them, since how far
    #: a driver sees round a bend is what decides whether a pass is on. On this
    #: landscape it buys nothing to pay for that: the same structures over the
    #: same share of the lap, the same earthwork, the same grade.
    ease: bool = False
    #: How the trees are carried: 'field' (a table and its species) or 'tiles'.
    forest: str = 'field'
    #: Where the species' files are; None for the shipped ones.
    species_directory: str | None = None
    #: Whether the world carries zones for the places its road runs through --
    #: its bores, causeways, bridges and wooded stretches -- each lit by an
    #: environment captured in it and heard with its own ambience
    #: (:meth:`zone_layer`).
    places: bool = True
    wetness: float = 0.0
    _circuit: RoadPath | None = field(default=None, init=False, repr=False)
    _terrain: Layer | None = field(default=None, init=False, repr=False)
    _landscape: FieldTerrainLayer | None = field(default=None, init=False,
                                                 repr=False)
    _scatter: Any = field(default=None, init=False, repr=False)
    _character: Any = field(default=None, init=False, repr=False)
    _start_line: StartFinish | None = field(default=None, init=False,
                                            repr=False)

    def natural(self) -> HeightFn:
        """The land before the road touched it: the height source, composed.

        A world given no source is the shipped landscape at its own ``relief``,
        which is the same thing said the short way.
        """
        return self.height_source().height_fn()

    def height_source(self) -> HeightSource:
        """Where the ground comes from, whether it was given one or not."""
        if self.source is not None:
            return self.source
        return HeightSource(base=ProceduralBase(relief=self.relief))

    def treeline(self) -> tuple[float, float]:
        """The band trees grow in: from this world's waterline to its treeline.

        The floor is :attr:`water_level` and not a number written down, because
        a world that floods its valleys has its forest on the islands. Taken
        from the default instead, a raised sea leaves three quarters of the
        trees standing on the sea bed. The ceiling scales with the world's own
        relief.
        """
        return (self.water_level + TREE_FREEBOARD, TREELINE * self.relief)

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
        layers: list[Layer] = [self.terrain()]
        if self.landscape() is not self.terrain():
            # The ground is meshed into the tiles, and the landscape goes with
            # it: what the tiles are blended and lit from, and the surface the
            # world is collided against, walked on and planted on.
            layers.append(self.landscape())
        layers.extend([self.water(), self.trees()])
        rivers = self.rivers()
        if rivers is not None:
            layers.append(rivers)
        if self.road:
            layers.append(self.circuit_layer())
            signs = self.sign_layer()
            if signs is not None:
                layers.append(signs)
            layers.append(self.gantry_layer())
        props = self.prop_layer()
        if props is not None:
            layers.append(props)
        stones = self.stone_layer()
        if stones is not None:
            layers.append(stones)
        if self.road and self.places:
            layers.append(self.zone_layer())
        return layers

    def zone_layer(self) -> ZonesLayer:
        """The zones of the places the circuit runs through.

        One per piece of every bore, causeway and bridge, and of every stretch
        of plain road with the forest close beside it
        (:func:`~OpenGLContext_editor.world.places.road_places`), each with
        what that kind of place holds: an environment captured on the road
        inside it, and birdsong in the forest, surf at a causeway, or a bore's
        reverb (:data:`~OpenGLContext_editor.bake.zones.PLACE_SETTINGS`).
        """
        tunnel = self.tunnel_profile()
        road = self.circuit()
        trees = self.scatter().positions if self.tree_density > 0.0 else None
        places = road_places(
            road, trees,
            tunnel_half_width=road.profile.on_structure().total_width / 2.0
            + tunnel.margin,
            tunnel_height=tunnel.clearance)
        return ZonesLayer(zone_records(places), place_sounds(self.seed))

    def water(self) -> Layer:
        """The lakes: a sheet wherever the ground dips below the waterline.

        Its own surface rather than the ground clamped flat, which is what
        gives the world a shoreline -- the shore is where the land passes
        through the water, and there is no such line on a ground that has been
        levelled at it.
        """
        return WaterLayer(height_fn=self.height_fn(), extent=self.footprint(),
                          level=self.water_level, name='water')

    def rivers(self) -> Any:
        """The water running down this world's channels, or None if it has none.

        Given the land **without** the channels in it: the surface is measured
        down from the ground the beds were cut into, which is what puts water
        in a valley rather than a ribbon on a hillside.
        """
        if not len(self.channels):
            return None
        from OpenGLContext_editor.bake.rivers import RiverLayer
        return RiverLayer(channels=list(self.channels),
                          ground=self.height_source().height_fn(),
                          name='river')

    def rocks(self) -> Any:
        """Where the boulders lie: near the road, clear of the carriageway.

        Near it because that is what makes a rock an obstacle rather than
        scenery: it is the thing a car leaving the road meets. Clear of the
        carriageway because an obstacle a driver cannot avoid is not an
        obstacle, it is a wall.
        """
        from OpenGLContext.loaders.tiles3d.scatter import Scatter
        placed = scatter_on_heightfield(
            self.seated_on(), self.footprint(), density=ROCK_DENSITY,
            seed=self.seed + 101, scale_range=ROCK_RADIUS,
            slope_limit=ROCK_SLOPE_LIMIT, slope_fn=self.slope_fn(),
            height_range=(self.water_level + ROCK_FREEBOARD, 1.0e9),
            keep=self._beside_the_road)
        return Scatter(placed.positions, placed.yaws, placed.scales)

    def stones(self) -> Any:
        """The loose stone on the hillsides: everywhere but on the road.

        Seated on the landscape's own surface (:meth:`seated_on`) -- grain
        included, since the landscape carries the grain the tiles draw. So a
        stone lies on the hillside in the picture and on the hillside
        underfoot, from the one number.

        Kept out of the strip that was cleared to build the road, the same
        strip the trees and the boulders are kept out of: a machine went
        through it and nothing on it is still standing. Kept off nothing else,
        because a landscape has stone in it.
        """
        from OpenGLContext.loaders.tiles3d.scatter import Scatter
        placed = scatter_on_heightfield(
            self.seated_on(), self.footprint(), density=STONE_DENSITY,
            seed=self.seed + 307, scale_range=STONE_RADIUS,
            slope_limit=STONE_SLOPE_LIMIT, slope_fn=self.slope_fn(),
            height_range=(self.water_level + ROCK_FREEBOARD, 1.0e9),
            keep=self._off_the_road)
        return Scatter(placed.positions, placed.yaws, placed.scales)

    def seated_on(self) -> Any:
        """The surface anything standing on the ground is placed on.

        The landscape's own grid, which is what the world is collided against
        and clamped to -- not the height function it was sampled from. The two
        differ by however much a cell's interpolation differs from the function
        across it, which on wide cells over real relief is a boulder floating
        half a metre over the hillside it is lying on.
        """
        ground = self.landscape().field()
        return ground.sample

    def stone_shapes(self) -> dict:
        """The handful of stones this world's loose rock is cut from."""
        return {_stone_kind(index): rock_mesh(radius=1.0, seed=100 + index,
                                              profile=STONE_PROFILE)
                for index in range(STONE_SHAPES)}

    def stone_layer(self) -> Any:
        """The loose stone, or None for a world with none to strew.

        Each stone is bedded into the ground by a share of its own size, so it
        lies in the hillside rather than resting on it at a point, and the
        bedded foot is what is both drawn and stood on.
        """
        placed = self.stones()
        if not len(placed.positions):
            return None
        from OpenGLContext_editor.bake.stones import StoneLayer
        prototypes = self.stone_shapes()
        stones = [
            Prop.of(prototypes[_stone_kind(index % STONE_SHAPES)],
                    kind=_stone_kind(index % STONE_SHAPES),
                    position=(point[0], point[1] - STONE_SETTLE * float(scale),
                              point[2]),
                    yaw=float(yaw), scale=float(scale), shape='dome')
            for index, (point, yaw, scale) in enumerate(
                zip(placed.positions, placed.yaws, placed.scales, strict=True))]
        return StoneLayer(stones=stones, prototypes=prototypes, name='stones')

    def _off_the_road(self, points: np.ndarray) -> np.ndarray:
        """Which placements are outside the strip cleared for the road."""
        return self.outside_the_clearing(
            points, margin=STONE_RADIUS[1] + STONE_CLEARANCE)

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

        Not inside a bore: the ground a boulder was lying on is the ground that
        came out to make the tunnel, and one left behind is a rock on the floor
        of it -- lit through the portal, and solid to hit.
        """
        if not self.road:
            return np.ones(len(points), dtype=bool)
        circuit = self.circuit()
        found = circuit.sample(points[:, 0], points[:, 2],
                               radius=ROCK_REACH * 1.5)
        keep = self.outside_the_clearing(
            points, margin=ROCK_RADIUS[1] + ROCK_CLEARANCE)
        keep = keep & np.asarray(found.distance < ROCK_REACH)
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
        """The circuit's signs, or None for a world with no road.

        Nothing here decides what the warnings say: the alignment does, from its
        own curvature, its own grade and the structures along it, and how fast
        each bend is worth from its own radius. The posted limit is the one
        thing a road cannot work out about itself, so it is told
        (:attr:`posted`). See :mod:`OpenGLContext_editor.world.signs`.
        """
        if not self.road:
            return None
        circuit = self.circuit()
        warnings = warn_of(circuit, CIRCUIT_DESIGN_SPEED, limit=self.posted)
        if not warnings:
            return None
        profile = SignProfile(offset=SIGN_OFFSET)
        return SignLayer(sign_placements(circuit, warnings, profile=profile,
                                         ground=self.height_fn()),
                         profile=profile)

    def start_station(self) -> float:
        """How far along the circuit a lap begins, in metres.

        Zero unless a start point was chosen, in which case it is the station
        nearest that point. Nearest rather than an index into the drawn plan,
        because the centreline the game gets is not the plan: it has been
        eased, draped over the ground and re-sampled, and the point the
        designer put the line on is the thing that survives all of that.
        """
        if self.start_at is None:
            return 0.0
        path = self.circuit()
        where = np.asarray(self.start_at, dtype='d').ravel()
        found = path.sample(np.asarray([float(where[0])]),
                            np.asarray([float(where[-1])]))
        stations = np.asarray(path.stations, dtype='d')
        index = int(np.clip(int(found.segment.ravel()[0]), 0,
                            len(stations) - 1))
        return float(stations[index])

    def start_line(self) -> StartFinish:
        """Where the circuit's start/finish gantry stands.

        At :meth:`start_station`, which is where a lap begins and ends. A road
        that does not return to its start is marked at the point it sets off
        from unless the designer said otherwise.
        """
        if self._start_line is None:
            self._start_line = start_finish(self.circuit(),
                                            profile=GantryProfile(),
                                            ground=self.height_fn(),
                                            station=self.start_station())
        return self._start_line

    def gantry_layer(self) -> GantryLayer:
        """The circuit's start/finish marker: a beam over the road and a line
        painted under it."""
        return GantryLayer(placement=self.start_line())

    def height_fn(self) -> Any:
        """The ground as the world finally has it, earthworks included."""
        if not self.road:
            return self.natural()
        return conform_terrain(self.natural(), self.circuit(),
                               tunnel=self.tunnel_profile())

    def circuit(self) -> RoadPath:
        """The race circuit: laid out on the natural ground, smoothed, and told
        which of its stretches are carried rather than laid."""
        if self._circuit is None:
            ground = self.natural()
            plan = (np.asarray(self.route, dtype='d') if self.route is not None
                    else circuit_plan(self.extent * 0.32, self.extent * 0.25,
                                      variation=self.variety, seed=self.seed))
            # Corners are corners, drawn or generated: a vertex turns the road
            # through the whole of it at once, which no car can take. Each is
            # *rounded in place* rather than opened out, so a hairpin drawn to
            # climb a slope stays a hairpin and its legs stay where they were
            # put -- and the straights between them stay straight, which is
            # what a circuit is overtaken on.
            # Every corner rounded to the same radius is one corner repeated;
            # the world's *own* circuit draws each from a mix instead, so a lap
            # has somewhere to brake and somewhere to carry speed. The plan's
            # own turn angles are untouched -- what changes is how hard each
            # turn has to be taken.
            #
            # A route a caller gave is held to the one radius, because its
            # corners are already a decision somebody made: a designer who drew
            # a hairpin meant it, and re-drawing it from a mix is the generator
            # overruling them. What a drawn route still gets is everything
            # :meth:`character` derives -- how fast each stretch is for, where
            # it may climb, where it is left rough, where it is opened out --
            # all of which follows from the line they drew rather than replacing
            # it.
            drawn = self.route is not None
            plan = hold_corners(
                plan,
                minimum=(self.corner_radius() if drawn else
                         corner_radii(plan, self.corner_radius(),
                                      spread=self.variety, seed=self.seed)),
                closed=self.closed, spacing=CIRCUIT_SPACING)
            if self.ease and self.route is None:
                # And then slid onto ground it can follow, which is worth
                # doing to a road that is already drivable rather than
                # instead of making it one.
                plan = ease_route(
                    plan, ground, reach=CIRCUIT_REACH, rounds=CIRCUIT_EASING,
                    closed=self.closed, spacing=CIRCUIT_SPACING,
                    minimum_radius=self.corner_radius())
            character = self._character = self.character(plan, ground)
            line = follow_terrain(plan, ground, spacing=CIRCUIT_SPACING,
                                  smoothing=character.smoothing,
                                  maximum_grade=character.grade_limit,
                                  design_speed=character.design_speed,
                                  minimum_height=(self.water_level
                                                  + CAUSEWAY_FREEBOARD),
                                  closed=self.closed)
            self._circuit = RoadPath(
                line, profile=CIRCUIT_PROFILE, ops=self._ops(line),
                bank=bank_profile(line, CIRCUIT_DESIGN_SPEED,
                                  profile=CIRCUIT_PROFILE,
                                  maximum=self.maximum_bank,
                                  closed=self.closed),
                # The corridor opens out with the carriageway: a lane added
                # inside the strip that was cleared for two runs into the trees.
                clearance=character.clearance + character.widening / 2.0,
                widening=character.widening)
        return self._circuit

    def circuit_character(self) -> Any:
        """What kind of road each stretch of the built circuit turned out to be.

        The design speed, grade limit, smoothing and cleared width the alignment
        was actually settled with, one figure per point of it. A road that
        varies cannot be checked against a single figure -- the question "does a
        car stay on it" is asked of each stretch at the speed *that* stretch is
        for -- and this is what answers it.
        """
        self.circuit()
        return self._character

    def character(self, plan: Any, ground: Any) -> Any:
        """What kind of road each stretch of this circuit is to be.

        The design speed, the grade limit, the smoothing and the cleared width,
        one figure per point of the alignment about to be built from ``plan``
        (:func:`~OpenGLContext_editor.world.character.road_character`). At
        :attr:`variety` 0 all four come out as the world's own single figures,
        which is the road this generator used to build.
        """
        if self.variety <= 0.0:
            return road_character(
                plan, ground, design_speed=CIRCUIT_DESIGN_SPEED,
                spacing=CIRCUIT_SPACING, closed=self.closed,
                grade_limit=CIRCUIT_MAX_GRADE, steep_grade=CIRCUIT_MAX_GRADE,
                smoothing=CIRCUIT_SMOOTHING, least_smoothing=CIRCUIT_SMOOTHING,
                clearance=self._corridor(), most_clearing=self._corridor())
        return road_character(
            plan, ground, design_speed=CIRCUIT_DESIGN_SPEED,
            spacing=CIRCUIT_SPACING, closed=self.closed,
            grade_limit=CIRCUIT_MAX_GRADE, steep_grade=CIRCUIT_STEEP_GRADE,
            smoothing=CIRCUIT_SMOOTHING,
            clearance=self._corridor(),
            most_clearing=self._corridor() + MOST_CLEARING_BEYOND,
            climbing_lane=CLIMBING_LANE)

    def corner_radius(self) -> float:
        """The tightest corner this world's circuit may have, in metres.

        What the design speed asks for once the lean is allowed for
        (:func:`~OpenGLContext.scenegraph.road.cornering_radius`): a corner
        banked to :attr:`maximum_bank` holds the speed at a radius a flat one
        could not, and holding the plan to the flat figure anyway would throw
        the whole of what the banking buys away.

        It is a **floor on the corner and so on the speed**: nothing tighter is
        laid, and every corner looser than it -- which is most of them -- holds
        appreciably more than the design speed rather than exactly it. The floor
        is kept by :data:`CORNER_MARGIN`, so that what is *built* clears it
        rather than sitting exactly on it.

        A plan whose legs are too short to fit the fillet keeps its corner and
        loses radius instead
        (:func:`~OpenGLContext_editor.world.route.hold_corners`) -- a drawn
        hairpin stays a hairpin. Such a corner is slower than the design speed,
        and the road says so: what it is signed at comes from the radius and the
        lean it ended up with.
        """
        return CORNER_MARGIN * cornering_radius(CIRCUIT_DESIGN_SPEED,
                                                bank=self.maximum_bank)

    def _ops(self, line: np.ndarray) -> Any:
        """What is built along the alignment, point by point."""
        if not self.structures:
            return None
        natural = np.asarray(self.natural()(line[:, 0], line[:, 2]), dtype='d')
        chosen = choose_structures(line, natural, waterline=self.water_level,
                                   closed=self.closed)
        ops = np.full(len(line), Op.DIRT, dtype=object)
        for structure in chosen:
            ops[structure.indices(len(line))] = structure.kind
        return ops

    def circuit_layer(self) -> RoadLayer:
        return RoadLayer(self.circuit(), wetness=self.wetness,
                         ground=self.natural(), shade=self.canopy_shade(),
                         start=self.start_station(), posted=self.posted,
                         tunnel=self.tunnel_profile())

    def ground_spacing(self) -> float:
        """How far apart the ground is sampled, in metres.

        The field's own spacing where the world carries one, and the *finest
        tile's* where it tiles its ground instead -- which is the resolution
        anything cut out of the ground is cut at. A tiled world's finest tile
        covers ``extent / 2**depth`` metres at ``resolution`` samples, so the
        depth of the tree is half the answer: measured against the root tile
        instead, a portal's face comes out sixty metres across and the ground
        cleared in front of it is a trench down the approach.
        """
        if self.ground == 'field':
            return self.extent / max(self.field_resolution - 1, 1)
        return (self.extent / float(1 << max(int(self.depth), 0))
                / max(self.resolution - 1, 1))

    def tunnel_profile(self) -> TunnelProfile:
        """The bore this world builds, with a face wide enough for its ground.

        A portal's face is what covers the edge of the hole its mouth is cut in
        (:func:`~OpenGLContext.scenegraph.roadworks.bore_opening`), and the hole
        is cut on the ground's own grid: a triangle the opening reaches into
        goes whole, which takes ground most of a cell past it. A face narrower
        than that leaves daylight down each side of the portal, so on coarse
        ground the border is as wide as a cell rather than as wide as it looks
        best.
        """
        border = max(TunnelProfile().portal_border, self.ground_spacing())
        return TunnelProfile(portal_border=border)

    def terrain(self) -> Layer:
        """The ground as it is *drawn*: the landscape itself, or the tiles.

        One layer either way, and for a world whose ground is its field it is
        the same object :meth:`landscape` answers.
        """
        if self._terrain is None:
            self._terrain = self._build_terrain()
        return self._terrain

    def field_spacing(self) -> float:
        """How far apart the landscape's own height grid is sampled, in metres.

        The landscape is what the world is *collided* against, walked on and
        planted on, so this is the finest detail a world can be made to feel --
        whatever its tiles draw.
        """
        return self.extent / max(self.field_resolution - 1, 1)

    def detail_error(self) -> float:
        """The geometric error of the finest tile this world will be baked to.

        The root's error halves at every level
        (:func:`~OpenGLContext_editor.bake.driver.root_error_for`), and the
        finest tile is the one whose surface has to agree with the landscape,
        so its error is what the grain in both is held inside.
        """
        return root_error_for(self.extent, self.resolution) \
            / float(1 << max(int(self.depth), 0))

    def grain_drawn(self) -> Relief | None:
        """The grain the tiles carry, held to what the landscape can hold too.

        None for a world with no grain and for one that draws its own field,
        which has no tiles to put detail in.

        A band finer than the landscape's grid is relief a player would see and
        walk straight through, so it is cut from what is drawn as well
        (:meth:`~OpenGLContext.scenegraph.terrain.Relief.no_finer_than`). What
        is left, the landscape carries: :meth:`field_terrain` builds it with the
        same grain at the same figures, so the finest tile and the surface under
        it are one surface.
        """
        if self.grain is None or self.ground != 'tiles':
            return None
        held = self.grain.no_finer_than(self.field_spacing())
        return replace(held, where=self.grain_applies())

    def grain_applies(self) -> Any:
        """How much grain each place gets, from 0 on the road to 1 clear of it.

        A road is built by *clearing and levelling* a strip of ground, so the
        grain stops at the edge of that strip (:meth:`corridor_along`) and not
        at the edge of the tarmac -- the same strip the trees, the boulders and
        the loose stone are kept off. Grain inside it is ground standing up
        where a machine levelled it, and where a car that has run wide is
        trying to recover. It comes back over :data:`GRAIN_FADE` metres beyond
        the clearing, because a grader leaves a batter and not a cliff.

        Ground the road *passes* rather than sits on keeps its grain: the
        hillside a bore runs under, the valley a deck crosses. What separates
        them is height, not which structure carries the road -- the ground at a
        portal and at a deck's abutment comes up to meet the carriageway
        whatever the stretch is called, and that is exactly where a hummock
        would stand in the way. So the fade is full where the ground is at the
        road's own level and gone by :data:`GRAIN_ABOVE` metres clear of it.
        """
        if not self.road:
            return None
        circuit = self.circuit()
        graded = self.corridor_along()
        ground = self.height_fn()
        radius = (float(graded.max()) + GRAIN_FADE) * 1.5

        def applies(x: Any, z: Any) -> Any:
            found = circuit.sample(x, z, radius=radius)
            away = np.asarray(found.distance, dtype='d')
            reached = np.isfinite(away)
            edge = graded[found.segment]
            out = np.clip((away - edge) / GRAIN_FADE, 0.0, 1.0)
            rise = np.abs(np.asarray(ground(x, z), dtype='d')
                          - np.where(reached, found.height, 0.0))
            beside = np.clip((GRAIN_ABOVE - rise)
                             / (GRAIN_ABOVE - GRAIN_AT_GRADE), 0.0, 1.0)
            worked = np.where(reached, (1.0 - out) * beside, 0.0)
            return 1.0 - worked
        return applies

    def detailed(self, ground: HeightFn) -> HeightFn:
        """``ground`` with this world's grain in it, at the finest tile's figures.

        What the landscape is sampled from and what anything standing on the
        ground is seated on. Fixed figures rather than per-tile ones: a tile
        picks its bands by its own spacing because that is level of detail, and
        the surface everything else agrees about has one answer.
        """
        grain = self.grain_drawn()
        if grain is None:
            return ground
        return grain.over(ground, spacing=self.ground_spacing(),
                          error=self.detail_error())

    def landscape(self) -> FieldTerrainLayer:
        """The landscape as a height image and a control map.

        Written beside the tileset whichever way the ground is drawn. It is what
        the ground is blended from and lit by, and the surface a game collides
        against, clamps a camera to and seats a plant on -- which is the surface
        that must not change resolution under a wheel as a tile refines.
        """
        if self._landscape is None:
            self._landscape = self.field_terrain()
        return self._landscape

    def _build_terrain(self) -> Layer:
        if self.ground == 'field':
            return self.landscape()
        if self.ground != 'tiles':
            raise ValueError("a world's ground is 'field' or 'tiles', not %r"
                             % (self.ground,))
        return HeightfieldLayer(
            height_fn=self.height_fn(), height_fn_at=self.height_fn_at(),
            extent=self.footprint(), resolution=self.resolution,
            holes=self.bore_openings(),
            # No clamp: the ground is meshed as it is and the water is laid
            # over it, so the shoreline is where the land actually passes
            # through the surface.
            color_fn=terrain_colors, water_level=None,
            relief=self.grain_drawn(), finest_error=self.detail_error(),
            name='terrain')

    def bore_openings(self) -> Any:
        """Where a bore's mouth breaks the ground, or None if none does.

        A hill a road runs inside is drawn as a hill, which at the portal puts
        the hillside where the carriageway is. This is what comes out of it
        (:meth:`OpenGLContext_editor.world.road.RoadPath.bore_openings`), cut
        into the tiles at bake time where the tiles are what is drawn.

        The surface it is measured against is the conformed ground -- the land
        with the road's own earthworks already in it -- because that is the
        surface being cut.
        """
        if not self.road:
            return None
        return self.circuit().bore_openings(
            self.height_fn(), tunnel=self.tunnel_profile(),
            approach=BORE_APPROACH_CELLS * self.ground_spacing())

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
        field = self.landscape().field()

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
        if self.forest != 'field':
            return None
        return SplatTerrain(self.landscape().field(), list(GROUND_LAYERS),
                            control=None,
                            canopy=self.scatter().positions).shade

    def field_terrain(self) -> FieldTerrainLayer:
        """The landscape as one height field and one splat control map."""
        at_spacing = self.height_fn_at()
        return FieldTerrainLayer(
            height_fn=self.detailed(self.height_fn()),
            height_fn_at=(None if at_spacing is None
                          else lambda spacing: self.detailed(at_spacing(spacing))),
            extent=self.footprint(), resolution=self.field_resolution,
            control_size=self.control_size,
            layers=list(GROUND_LAYERS),
            rules=list(ground_rules(self.water_level)),
            road=self.circuit() if self.road else None,
            road_layer=GROUND_LAYERS.index('dirt'),
            road_corridor=self._corridor(),
            drawn=self.ground,
            name='terrain')

    def _corridor(self) -> float:
        """How far out the road's own ground reaches, in metres.

        Out to where the trees start, so the bare strip beside the carriageway
        is the strip that was cleared for it and the forest floor begins where
        the forest does.
        """
        across: float = CIRCUIT_PROFILE.total_width
        return across / 2.0 + ROAD_CLEARANCE

    def widest_corridor(self) -> float:
        """The widest the road's own ground gets, in metres from the centreline.

        What a splat map or a bare strip has to be painted out to: the corridor
        is not one width any more, and painting it at the narrow one leaves the
        ground cover growing over the extra lane on every climb.
        """
        if not self.road:
            return self._corridor()
        beside = self.circuit().clearance
        return (self._corridor() if beside is None
                else float(np.max(beside)))

    def height_fn_at(self) -> Any:
        """The ground as a function of the spacing a tile samples it at.

        A cutting narrower than a coarse tile's vertex spacing would be stepped
        over and the road in it buried, so each tile gets the earthwork widened
        to its own resolution.
        """
        if not self.road:
            return None
        return conform_terrain_at(self.natural(), self.circuit(),
                                  tunnel=self.tunnel_profile())

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

        The kinds of plant rather than a scatter of them: where each one stands
        is worked out as the camera moves, not baked -- see
        :class:`~OpenGLContext.scenegraph.vegetation.cover.GroundCover`. They
        come from the same place the tree species do, so a world pointed at its
        own art gets its own undergrowth.
        """
        try:
            return shipped_cover(self.species_directory)
        except LookupError:
            return None

    def bake(self, directory: str, **named: Any) -> BakeResult:
        """The world baked into ``directory`` as 3D Tiles, at its own depth.

        :func:`~OpenGLContext_editor.bake.driver.bake_world` over
        :meth:`layers`, with :attr:`depth` and :meth:`credits`; ``named`` is
        any other ``bake_world`` option. A ``depth`` other than the world's is
        refused, since the world's ground spacing, portals and finest grain
        are measured against a cell of that depth.
        """
        depth = named.pop('depth', self.depth)
        if int(depth) != int(self.depth):
            raise ValueError(
                'this world is laid out for a tree of depth %d, not %d; give '
                'the world the depth to bake at' % (self.depth, depth))
        named.setdefault('credits', self.credits())
        return bake_world(self.layers(), directory, depth=int(self.depth),
                          **named)

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

    def outside_the_clearing(self, points: np.ndarray, margin: float = 0.0,
                             carried: float = 0.0) -> np.ndarray:
        """Which placements are outside the strip cleared for the road.

        A road is not only a surface: it is a strip of ground that was cleared
        to build it, and nothing a machine went through is still standing on
        it. One answer for everything scattered -- trees, boulders, loose stone
        -- so a world does not carry three ideas of where its road is.

        The corridor is the road's own (:meth:`_corridor`), opened out where a
        driver has to see round the bend they are on, so a fast corner is a
        clearing and the straight after it is a road through trees. Over a bore
        it is the whole of what was dug out
        (:meth:`~OpenGLContext_editor.world.road.RoadPath.reshaped_segments`):
        the land is not there any more, and something standing on the floor of
        the cut is inside the tunnel, in plain view through the portal.

        ``margin`` is how much room the thing being placed needs of its own --
        a boulder's own radius, say. ``carried`` is extra room where the road
        stands above the land the thing is rooted in: the crown of a tree at
        the foot of an embankment grows through the side of it rather than over
        the carriageway.
        """
        if not self.road:
            return np.ones(len(points), dtype=bool)
        circuit = self.circuit()
        corridor = self.corridor_along() + float(margin)
        reach = corridor + float(carried)
        found = circuit.sample(points[:, 0], points[:, 2],
                               radius=reach.max() * 1.5)
        ground = np.asarray(self.natural()(points[:, 0], points[:, 2]),
                            dtype='d')
        above = found.height - ground > CARRIED_ABOVE
        segment = found.segment
        cleared = np.where(above, reach[segment], corridor[segment])
        return np.asarray(found.distance > cleared)

    def corridor_along(self) -> np.ndarray:
        """How far the clearing reaches from the centreline, per segment.

        The road's own corridor, opened out where a driver has to see round the
        bend they are on and wider again over a bore, where the whole of what
        was dug out is corridor. This is the strip a machine went through: what
        is scattered is kept off it (:meth:`outside_the_clearing`) and the
        ground's own grain stops at it (:meth:`grain_applies`), because ground
        that was cleared and levelled has no hummocks left in it.
        """
        circuit = self.circuit()
        along = circuit.clearance_along()
        if along is None:
            return np.full(len(circuit.ops) - 1, self._corridor())
        return np.asarray(along, dtype='d')

    def _away_from_the_road(self, points: np.ndarray) -> np.ndarray:
        """Which placements leave the cleared corridor room for a crown."""
        return self.outside_the_clearing(points, carried=CROWN_RADIUS)



def _rock_kind(index: int) -> str:
    """The name of one of the cut boulders, as a prop's kind."""
    return 'rock%d' % (index,)


def _stone_kind(index: int) -> str:
    """The name of one of the cut stones, as a prop's kind."""
    return 'stone%d' % (index,)


def _slopes(height_fn: Any, positions: np.ndarray, step: float = 8.0
            ) -> np.ndarray:
    """How steep the ground is under each placement, as rise over run."""
    x, z = positions[:, 0], positions[:, 2]
    dx = (np.asarray(height_fn(x + step, z), dtype='d')
          - np.asarray(height_fn(x - step, z), dtype='d')) / (2.0 * step)
    dz = (np.asarray(height_fn(x, z + step), dtype='d')
          - np.asarray(height_fn(x, z - step), dtype='d')) / (2.0 * step)
    return np.hypot(dx, dz)


def circuit_plan(radius_x: float, radius_z: float,
                 harmonics: Sequence[tuple[int, float]] = CIRCUIT_HARMONICS,
                 corners: int = CIRCUIT_CORNERS,
                 variation: float = 0.0, seed: int = 0) -> np.ndarray:
    """A closed race circuit as a plan **as drawn**: one point per corner.

    Straights joined by corners, which is what a circuit is.

    ``corners`` points are placed around an ellipse whose radius is modulated
    by a few harmonics of the angle, so no two are the same distance out and
    the circuit is not a regular polygon. What comes back is those points and
    nothing between them: a vertex is a corner, which is what
    :func:`~OpenGLContext_editor.world.route.hold_corners` reads a plan as, and
    it rounds each to a radius a car can take using the whole length of the
    legs either side. Handed a plan already re-sampled along its straights it
    has millimetres of leg to work with and leaves the corner where it was.

    **Straights are what a circuit is passed on.** How far a driver can see
    round a bend of radius *r* is `sqrt(8 * r * clear)`, and a road cut through
    a wood offers a hundred-odd metres of that against the two hundred an
    overtake at racing speed needs -- so a circuit that is one continuous bend
    is a circuit nobody overtakes on, and the traffic on it is scenery to queue
    behind. A plan of straights has somewhere to do it.

    ``variation`` from 0 to 1 is **how unlike each other the corners are**. At 0
    the vertices are evenly spaced round the ellipse and every corner turns
    through much the same angle over much the same length of leg, which is one
    corner repeated: a driver who has taken the first has taken them all. Turned
    up, the *angles between* the vertices and their distances out are drawn
    apart, so the lap gets a long straight somewhere and a pair of corners in
    quick succession elsewhere, and the corners themselves turn through
    anything from a kink to most of a reversal. ``seed`` picks which lap; the
    same seed is always the same lap.

    What comes back is still a plan -- vertices and nothing between them -- so
    how *tight* each of those corners is remains
    :func:`~OpenGLContext_editor.world.route.hold_corners`'s to say, and it
    takes a radius per corner.
    """
    count = max(int(corners), 3)
    spread = float(np.clip(variation, 0.0, 1.0))
    turn = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    if spread > 0.0:
        turn = _uneven_turns(count, spread, seed)
    radius = np.ones_like(turn)
    for order, amount in harmonics:
        radius = radius + amount * np.sin(order * turn + order)
    if spread > 0.0:
        # Vertices drawn in and out as well as round: a vertex well inside its
        # two neighbours turns the road much further than one in line with
        # them, which is what puts a hairpin on a lap of sweepers.
        draw = np.random.default_rng(seed + 1).uniform(-1.0, 1.0, count)
        radius = radius * (1.0 + spread * CIRCUIT_EXCURSION * draw)
    apex: np.ndarray = np.stack([radius_x * radius * np.cos(turn),
                                 radius_z * radius * np.sin(turn)], axis=-1)
    return apex


def _uneven_turns(count: int, spread: float, seed: int) -> np.ndarray:
    """Angles round the circuit, drawn apart but still going once round.

    The *steps* between vertices are what is varied, and then normalised back
    to a full turn: a long step is a straight and two short ones together are a
    pair of corners a driver has no time between. Held above
    :data:`CIRCUIT_LEAST_STEP` of an even share, because two vertices on top of
    each other are one corner drawn twice rather than a chicane.
    """
    least = CIRCUIT_LEAST_STEP
    drawn = np.random.default_rng(seed).uniform(least, 2.0 - least, count)
    steps = 1.0 + spread * (drawn - 1.0)
    steps = steps / steps.sum() * 2.0 * math.pi
    found: np.ndarray = np.concatenate([[0.0], np.cumsum(steps)[:-1]])
    return found


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
