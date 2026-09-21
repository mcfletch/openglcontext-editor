"""The bust gallery: a hall, plinths down it, and a bust on each.

The world the level-of-detail demo is set in. A plain room -- polished parquet,
white plaster walls, dark beams across the top of them and the sky above -- and
a hundred and twenty copies of one bust receding down it on rectangular
plinths.

The arrangement is the demonstration rather than the scenery. One bust shows
whether a decimator kept a chin; a hall of them is the only view in which every
level of a chain is on screen at once, the near ones at their finest and the far
ones at their coarsest, with the switch happening somewhere in the middle where
it can be watched. It is also the only view that exercises what a game pays for
a chain, because the copies share their levels: the busts drawing at a given
level collapse into one instanced draw, so the frame costs one draw per level in
use and not one per bust.

Nothing here imports Blender. The add-on asks where things go and then makes
them, which is also what lets the measurements be checked without opening a
window.

**Coordinates are Blender's**: X across the hall, Y along it, Z up, the origin
in the middle of the floor with the floor's top face at zero. The glTF exporter
converts to Y-up on the way out.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    'Gallery',
    'Placement',
    'Settling',
    'Slab',
    'Sun',
    'Viewpoint',
    'coverage_at',
]

#: The vertical field of view a viewer is assumed to have, in degrees. Screen
#: coverage is a share of the window's *height*, so this is what turns a
#: distance into one. OpenGLContext's own default, so the thresholds written
#: into the world are the thresholds it reads them back at.
FIELD_OF_VIEW = 60.0


def coverage_at(radius: float, distance: float,
                field_of_view: float = FIELD_OF_VIEW) -> float:
    """What share of the window's height a sphere spans, at most 1.

    The same arithmetic ``OpenGLContext.scenegraph.lod`` selects levels by, so a
    hall laid out against it is a hall whose busts switch where they were meant
    to.
    """
    if distance <= 0:
        return 1.0
    tangent = math.tan(math.radians(field_of_view) / 2.0)
    return min(1.0, radius / (distance * tangent))


@dataclass(frozen=True)
class Slab:
    """A rectangular box: the whole of the room's geometry."""

    name: str
    material: str
    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    #: Rotation about Z, in radians: how far out of square it stands.
    turn: float = 0.0
    #: Whether this box is drawn into the shadow maps. A floor says no: there
    #: is nothing under it to shadow, and a floor in the depth pass shadows
    #: itself along every grazing ray. It is lit and takes shadows either way.
    casts: bool = True


@dataclass(frozen=True)
class Placement:
    """Where one bust stands, and which chain its levels belong to."""

    group: str
    position: tuple[float, float, float]
    #: Rotation about Z, in radians: which way the bust looks.
    turn: float


def _bearing(stand: tuple[float, float, float],
             target: tuple[float, float]) -> float:
    """The turn that faces a camera at ``stand`` towards ``target``.

    Zero looks down the hall towards -Y, which is the turn a
    :class:`Viewpoint` states.
    """
    return math.atan2(target[0] - stand[0], stand[1] - target[1])


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """``vector`` at unit length, which is what a direction is."""
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector)    # type: ignore[return-value]


@dataclass(frozen=True)
class Sun:
    """A directional light: which way it travels, and how strong it is."""

    name: str
    #: The unit vector the light travels *along*, from the sky towards the
    #: floor, so a sun leaning to the right has a positive first component.
    direction: tuple[float, float, float]
    #: The illuminance it delivers, in lux, which is the unit
    #: ``KHR_lights_punctual`` states a directional light in. Blender measures a
    #: sun in watts per square metre instead, so the builder divides by
    #: :data:`~.scene.LUMENS_PER_WATT` on the way in.
    lux: float
    #: Linear RGB.
    colour: tuple[float, float, float] = (1.0, 0.96, 0.9)
    #: Whether this light gets a shadow map. A light standing in for what a
    #: room's own surfaces throw back has nothing to shadow, and says so here.
    casts: bool = True


#: Two axes shifted at once reach further than either alone, so each is kept
#: to this share of the asked-for distance and the pair stay inside it.
_ROOT_HALF = math.sqrt(0.5)


@dataclass(frozen=True)
class Settling:
    """How far off true one plinth and the bust on it ended up."""

    across: float
    along: float
    turn: float
    bust_across: float
    bust_along: float
    bust_turn: float


@dataclass(frozen=True)
class Viewpoint:
    name: str
    position: tuple[float, float, float]
    #: Rotation about Z, in radians, from looking down -Y.
    turn: float


@dataclass(frozen=True)
class Gallery:
    """The hall, and everything standing in it.

    Every measurement is a metre. The defaults are the demo's: a hall long
    enough that a bust at the far end covers a fortieth of the window's height
    and one at arm's length covers more than half of it, which is the whole of
    a six-level chain in a single view.
    """

    #: Across the hall, wall face to wall face.
    width: float = 10.0
    #: Along it.
    length: float = 48.0
    #: Floor to the top of the walls, where the beams cross.
    height: float = 4.5
    #: How thick the floor and the walls are built.
    thickness: float = 0.25

    #: How many plinths there are along the hall.
    bays: int = 30
    #: Where the rows of plinths stand, across the hall.
    rows: tuple[float, ...] = (-4.6, -1.8, 1.8, 4.6)

    plinth_width: float = 0.62
    plinth_height: float = 1.05

    #: The bust's bounding radius, which is what its coverage is judged by.
    #: ``marble_bust_01`` is 0.27 x 0.51 x 0.30 metres, and a reader sizes a
    #: level from half the diagonal of what its ``POSITION`` accessor declares.
    bust_radius: float = 0.33

    beam_width: float = 0.22
    beam_depth: float = 0.36

    #: How far a beam is set along the hall from the row of busts it shades.
    #: It buys the light back: the shorter that reach, the less of the sun's
    #: lean has to be spent travelling along the hall to make it, and what is
    #: left over leans across -- which is what lights a wall, a plinth's side
    #: and a face. Beams square over the bust rows would need the sun almost
    #: straight down the hall, and a hall lit that way has two dark walls.
    beam_lead: float = 0.27

    #: What the sun delivers, in lux: the illuminance the glTF carries and the
    #: engine's light meter reads. Under the six the meter reads as neutral
    #: (``loaders/gltf/scene.py``, ``_meter_exposure``), so the hall is exposed
    #: the same way whether or not a sky is drawn behind its walls. A world
    #: over that number renders correctly only where the meter is consulted,
    #: which is the black-background case alone. The sky's own light is not
    #: metered at all, which is the other reason to leave the sun room.
    #:
    #: Blender measures a sun in watts per square metre, and its glTF exporter
    #: writes that as lux at :data:`~.scene.LUMENS_PER_WATT`; the builder does
    #: the division, so this field stays in the unit the file will hold.
    sun_lux: float = 5.5

    #: How high the sun stands, said as what it does: the length of shadow a
    #: thing lays down as a share of its own height. A third is mid-morning --
    #: enough that a plinth's shadow reaches across the floor beside it and the
    #: beams stripe the wall, and not so much that the hall is more shadow than
    #: floor.
    shadow_ratio: float = 1.0 / 3.0

    #: How tall the bust is, which is what its face spans. ``marble_bust_01``
    #: is 0.27 x 0.51 x 0.30 metres.
    bust_height: float = 0.51

    #: Where on a bust the beam's shadow stops, as a share of its height. The
    #: shadow covers the head *above* this line, so a little under three
    #: quarters puts it across the eyes: the crown is under the beam, the face
    #: is in the sun, and the edge runs between them.
    shadow_share: float = 0.72

    #: How far a plinth stands from where the lattice put it, at most: a few
    #: millimetres, which is a floor that is not quite flat and a stone that
    #: was set down by hand. A hall of a hundred and twenty exactly-placed
    #: plinths reads as arithmetic, and the eye finds the grid before it finds
    #: the busts.
    plinth_shift: float = 0.004

    #: And how far out of square, in radians -- a fifth of a degree.
    plinth_turn: float = math.radians(0.2)

    #: The same for a bust on its plinth, which was set down by the same hands
    #: and is small enough to turn further without looking thrown.
    bust_shift: float = 0.003
    bust_turn: float = math.radians(1.5)

    #: What settles the shifts. The same seed builds the same hall, so a world
    #: rebuilt is the world that was captured.
    jitter_seed: int = 20260921

    #: What the room's surfaces are called; the add-on makes a material per name.
    floor_material: str = 'GalleryFloor'
    wall_material: str = 'GalleryWall'
    beam_material: str = 'GalleryBeam'
    plinth_material: str = 'GalleryPlinth'

    #: What a bust's chain is called, with its number.
    group_format: str = 'bust_%03d'

    def __post_init__(self) -> None:
        if self.bays < 1:
            raise ValueError('a gallery needs at least one bay')
        if not self.rows:
            raise ValueError('a gallery needs at least one row of plinths')

    @property
    def bay(self) -> float:
        """How far apart the bays are along the hall."""
        return self.length / self.bays

    @property
    def shadow_line(self) -> float:
        """Where a beam's shadow is wanted on a bust: the height its edge cuts.

        The shadow lies *above* the line -- the crown of the head is under the
        beam and everything below the line is in the sun -- so at
        :attr:`shadow_share` of the bust's height it crosses about the eyes. A
        shadow that clears the heads is a shadow on the floor, and one that
        reaches the chin is a bust in the dark.
        """
        return self.plinth_height + self.shadow_share * self.bust_height

    def sun_bearing(self) -> float:
        """Which way the sun comes from, as an angle off straight across.

        Not a taste: it is what puts the beams' shadows where they are wanted.
        A beam's shadow is a line running across the hall, so moving the sun
        *across* the hall slides that line along itself and changes nothing --
        only the along-the-hall lean moves it, by the drop from the beam to
        whatever it lands on times :attr:`shadow_ratio`. Ask for it to land on
        the next row of busts, at :attr:`shadow_line`, and the angle follows.

        What is aimed is the *lower edge* of the shadow. A beam is a wide
        caster and a face is a steep one, so with the sun high the band it
        throws is far taller than a bust: it runs off the top of the head into
        the air above. Where it *stops* is the only part of it a bust shows,
        and that edge is drawn by the beam's under-trailing corner -- the last
        thing a ray climbing towards the sun passes under.

        Ask for that edge at :attr:`shadow_line` and the lean follows. A hall
        whose beams cannot reach the next row -- too high a sun for the span,
        or too short a bay -- takes the furthest lean there is, which is
        straight along the hall, and the edge lands wherever that puts it.

        A hall whose beams are too low or a sun too high to reach the next row
        takes the furthest lean there is, which is straight along the hall.
        """
        drop = (self.height - self.beam_depth) - self.shadow_line
        reach = drop * self.shadow_ratio
        wanted = self.bay / 2.0 - self.beam_lead + self.beam_width / 2.0
        if reach <= wanted:
            return math.pi / 2.0
        return math.asin(wanted / reach)

    def _refuse_a_crowded_hall(self) -> None:
        if self.bay <= self.plinth_width:
            raise ValueError(
                'a %.1fm hall in %d bays leaves %.2fm a bay, which will not '
                'hold a %.2fm plinth'
                % (self.length, self.bays, self.bay, self.plinth_width)
            )

    def room(self) -> list[Slab]:
        """Floor and four walls, open to the sky.

        There is no roof: what is overhead is the panorama
        (:mod:`~.sky`), and the beams cross the hall under it. A hall that is
        lit from outside and roofed over is a hall in its own shadow, and the
        plaster that was up there hid the one thing a bare sun is worth having
        for -- the beams cutting it into stripes across the busts.

        The walls cast, because they are what the hall's own shade is made
        of: a beam's shadow that crosses a wall and carries on past the line
        where the wall's own shadow starts reads as a shadow hanging in the
        air. The floor does not -- there is nothing under it to shadow, and a
        floor in the depth pass shadows itself along every grazing ray.
        """
        half_x, half_y = self.width / 2, self.length / 2
        thick, ceiling = self.thickness, self.height
        outer_x = self.width + 2 * thick
        outer_y = self.length + 2 * thick
        return [
            Slab('Floor', self.floor_material,
                 (0.0, 0.0, -thick / 2), (self.width, self.length, thick),
                 casts=False),
            Slab('Wall_North', self.wall_material,
                 (0.0, half_y + thick / 2, ceiling / 2),
                 (outer_x, thick, ceiling)),
            Slab('Wall_South', self.wall_material,
                 (0.0, -half_y - thick / 2, ceiling / 2),
                 (outer_x, thick, ceiling)),
            Slab('Wall_East', self.wall_material,
                 (half_x + thick / 2, 0.0, ceiling / 2),
                 (thick, outer_y, ceiling)),
            Slab('Wall_West', self.wall_material,
                 (-half_x - thick / 2, 0.0, ceiling / 2),
                 (thick, outer_y, ceiling)),
        ]

    def beams(self) -> list[Slab]:
        """The dark beams across the hall, one to a bay.

        They sit :attr:`beam_lead` short of square over the bust rows, which
        is what lets the sun keep some of its lean across the hall while its
        shadows still reach the next row's heads.
        """
        top = self.height
        return [
            Slab('Beam_%03d' % (index,), self.beam_material,
                 (0.0, self._along(index) + self.bay - self.beam_lead,
                  top - self.beam_depth / 2),
                 (self.width + 2 * self.thickness, self.beam_width,
                  self.beam_depth))
            for index in range(self.bays)
        ]

    def plinths(self) -> list[Slab]:
        """A rectangular plinth under every bust, each a little off true.

        The lattice says where a plinth was meant to go;
        :meth:`_settlings` says where it actually stands, which is a few
        millimetres off and a fraction of a degree out of square. Identical
        placement is the one thing a hundred and twenty copies of one object
        cannot survive: the eye reads the grid, and then it reads a rendering.
        """
        self._refuse_a_crowded_hall()
        return [
            Slab('Plinth_%03d' % (number,), self.plinth_material,
                 (across + settled.across, along + settled.along,
                  self.plinth_height / 2),
                 (self.plinth_width, self.plinth_width, self.plinth_height),
                 turn=settled.turn)
            for number, ((across, along), settled)
            in enumerate(zip(self._places(), self._settlings(), strict=True))
        ]

    def busts(self) -> list[Placement]:
        """A bust on the top face of every plinth, looking at the walkway.

        It carries its plinth's own shift -- a bust standing where the lattice
        put it while the stone under it moved is a bust off its plinth -- and
        adds a smaller one of its own, plus a degree or so of turn. What that
        buys close up is that no two of them catch the light the same way.
        """
        self._refuse_a_crowded_hall()
        return [
            Placement(self.group_format % (number,),
                      (across + settled.across + settled.bust_across,
                       along + settled.along + settled.bust_along,
                       self.plinth_height),
                      self._facing(across) + settled.bust_turn)
            for number, ((across, along), settled)
            in enumerate(zip(self._places(), self._settlings(), strict=True))
        ]

    def _settlings(self) -> list[Settling]:
        """How far off true each stone in the hall stands, in order.

        One generator walked once, so plinth *n* and the bust on it are settled
        together and the same seed builds the same hall however often it is
        built. ``random`` rather than numpy: this is a handful of numbers, and
        its stream is stable across versions in a way numpy's legacy one is not
        promised to be.
        """
        generator = random.Random(self.jitter_seed)
        settlings = []
        for _ in self._places():
            settlings.append(Settling(
                across=generator.uniform(-1.0, 1.0) * self.plinth_shift * _ROOT_HALF,
                along=generator.uniform(-1.0, 1.0) * self.plinth_shift * _ROOT_HALF,
                turn=generator.uniform(-1.0, 1.0) * self.plinth_turn,
                bust_across=generator.uniform(-1.0, 1.0) * self.bust_shift * _ROOT_HALF,
                bust_along=generator.uniform(-1.0, 1.0) * self.bust_shift * _ROOT_HALF,
                bust_turn=generator.uniform(-1.0, 1.0) * self.bust_turn,
            ))
        return settlings

    def lights(self) -> list[Sun]:
        """One sun, leaning across the hall and a little along it.

        A sun lights the whole hall evenly however long it is, which a row of
        lamps cannot: a lamp's light falls off with the square of the distance,
        so a hall lit by lamps is lit in pools with dark between them, and the
        far end of a long one is dark whatever the lamps are worth.

        One, because two of equal strength answer each other's shadows into a
        grey with no direction in it, and a room reads as lit by a rig rather
        than by an afternoon. What fills the shadow here is the sky
        (:mod:`~.sky`), which arrives from every direction at once, is
        the colour of the weather rather than of a second sun, and casts
        nothing.

        Where it stands is settled by what its shadows are for, rather than
        chosen: :attr:`shadow_ratio` says how long a shadow is against what
        throws it, and :meth:`sun_bearing` leans it along the hall until the
        beams' shadows land across the busts' faces. Straight down would stand
        every bust on its own shadow.

        Its lux is what decides the hall's exposure: the sky's own contribution
        is not metered, so the sun stays under what the engine's meter reads as
        neutral (``loaders/gltf/scene.py``, ``_meter_exposure``).
        """
        bearing = self.sun_bearing()
        across = self.shadow_ratio * math.cos(bearing)
        along = self.shadow_ratio * math.sin(bearing)
        return [Sun('Sun', _unit((across, -along, -1.0)), lux=self.sun_lux)]

    def sun_elevation(self) -> float:
        """How high the sun stands, as the sky panorama has to draw it."""
        return math.asin(-self.lights()[0].direction[2])

    def sun_azimuth(self) -> float:
        """Which way the sun stands, as the sky panorama has to draw it.

        The compass angle the light comes *from*, so that the bright quarter of
        the sky and the direction the shadows fall are the same afternoon.
        """
        direction = self.lights()[0].direction
        return math.atan2(-direction[1], -direction[0])

    def cameras(self) -> list[Viewpoint]:
        """Where the demo starts, and where it is going.

        Two, because the thing being shown is the difference between them: from
        the end of the hall every level of the chain is on screen at once, and
        from arm's length one bust fills enough of the window to be at its
        finest. A viewer cycles them, and walking between them is the demo.
        """
        # A bust on an *inner* row, at the far end, stood in front of at arm's
        # length. The inner row is the one a walk can reach without leaving the
        # walkway: a viewer -- or a camera moving between these two -- travels
        # the length of the hall down the middle and stops in front of it,
        # rather than crossing a row of plinths to get there.
        inner = min(self.rows, key=abs)
        # By the lattice rather than by where a bust ended up standing: the
        # settling moves every one of them off its exact coordinate, and a
        # camera aimed with an equality test would find none of them.
        across, along = next(place for place in self._places()
                             if place[0] == inner)
        inward = 1.0 if across < 0 else -1.0
        stand = across + inward * 1.1
        return [
            Viewpoint('Gallery', (0.0, self.length / 2 - 2.0, 1.65), turn=0.0),
            # Turned a quarter about to face back across the walkway at it.
            Viewpoint('Bust', (stand, along, 1.45), turn=-inward * math.pi / 2),
            # And one at the shoulder of the nearest bust, looking the length
            # of the hall at the far corner: the whole room in the frame with
            # one bust a hand's breadth away, drawing at its finest level
            # while the far end draws at its coarsest. The view the demo is
            # about, in one still.
            Viewpoint('Shoulder', self.shoulder_stand,
                      turn=_bearing(self.shoulder_stand, self.far_corner)),
        ]

    @property
    def shoulder_stand(self) -> tuple[float, float, float]:
        """Where that third camera stands: beside the nearest inner-row bust,
        at eye height, a step inside the end of the hall. A viewer reading the
        file in Y-up sees it as ``(-2.07, 1.65, -22.41)``."""
        return (-2.07, self.length / 2 - 1.59, 1.65)

    @property
    def far_corner(self) -> tuple[float, float]:
        """The corner of the hall diagonally opposite the shoulder."""
        return (self.width / 2.0, -self.length / 2.0)

    def _places(self) -> list[tuple[float, float]]:
        """Every plinth's footing, row by row along the hall."""
        return [(across, self._along(index) + self.bay / 2)
                for across in self.rows
                for index in range(self.bays)]

    def _along(self, index: int) -> float:
        """Where bay ``index`` starts, measured along the hall."""
        return -self.length / 2 + index * self.bay

    def _facing(self, across: float) -> float:
        """Which way a bust at ``across`` looks: inward, towards the walkway."""
        return math.pi / 2 if across < 0 else -math.pi / 2


#: The rows and bays the demo ships with, as a sentence for the documentation.
def describe(plan: Gallery) -> str:
    """One line saying what a hall holds and what it spans."""
    near = coverage_at(plan.bust_radius, 1.0)
    far = coverage_at(plan.bust_radius, plan.length)
    return (
        '%d busts in %d rows of %d down a %.0fm hall; a bust covers %.0f%% of '
        'the window at a metre and %.1f%% at the far end'
        % (len(plan.rows) * plan.bays, len(plan.rows), plan.bays, plan.length,
           near * 100, far * 100)
    )
