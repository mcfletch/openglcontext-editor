"""The bust gallery: a hall, plinths down it, and a bust on each.

The world the level-of-detail demo is set in. A plain room -- polished parquet,
white plaster walls, dark beams with plaster between them -- and a hundred and
twenty copies of one bust receding down it on rectangular plinths.

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
from dataclasses import dataclass

__all__ = [
    'Gallery',
    'Light',
    'Placement',
    'Slab',
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


@dataclass(frozen=True)
class Placement:
    """Where one bust stands, and which chain its levels belong to."""

    group: str
    position: tuple[float, float, float]
    #: Rotation about Z, in radians: which way the bust looks.
    turn: float


@dataclass(frozen=True)
class Light:
    name: str
    position: tuple[float, float, float]
    #: Watts, as Blender measures a point light.
    power: float
    #: Linear RGB.
    colour: tuple[float, float, float] = (1.0, 0.96, 0.9)


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
    #: Floor to the underside of the plaster.
    height: float = 4.5
    #: How thick the floor, walls and ceiling are built.
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

    #: What one ceiling lamp draws, in watts, as Blender measures a point
    #: light. The glTF exporter turns that into candela at about 54 to the
    #: watt, so this is a gallery lamp rather than a floodlight.
    lamp_power: float = 60.0

    #: How many lamps there are. OpenGLContext's forward PBR pass binds eight
    #: lights and ignores the rest, so a ninth would not make the hall
    #: brighter -- it would be a lamp that is never switched on.
    lamps: int = 8

    #: What the room's surfaces are called; the add-on makes a material per name.
    floor_material: str = 'GalleryFloor'
    wall_material: str = 'GalleryWall'
    ceiling_material: str = 'GalleryCeiling'
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

    def _refuse_a_crowded_hall(self) -> None:
        if self.bay <= self.plinth_width:
            raise ValueError(
                'a %.1fm hall in %d bays leaves %.2fm a bay, which will not '
                'hold a %.2fm plinth'
                % (self.length, self.bays, self.bay, self.plinth_width)
            )

    def room(self) -> list[Slab]:
        """Floor, four walls and the plaster overhead."""
        half_x, half_y = self.width / 2, self.length / 2
        thick, ceiling = self.thickness, self.height
        outer_x = self.width + 2 * thick
        outer_y = self.length + 2 * thick
        return [
            Slab('Floor', self.floor_material,
                 (0.0, 0.0, -thick / 2), (self.width, self.length, thick)),
            Slab('Ceiling', self.ceiling_material,
                 (0.0, 0.0, ceiling + thick / 2), (self.width, self.length, thick)),
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

        They hang from the plaster rather than being let into it, so the
        ceiling reads as beams *and* plaster from below.
        """
        top = self.height
        return [
            Slab('Beam_%03d' % (index,), self.beam_material,
                 (0.0, self._along(index), top - self.beam_depth / 2),
                 (self.width + 2 * self.thickness, self.beam_width,
                  self.beam_depth))
            for index in range(self.bays)
        ]

    def plinths(self) -> list[Slab]:
        """A rectangular plinth under every bust."""
        self._refuse_a_crowded_hall()
        return [
            Slab('Plinth_%03d' % (number,), self.plinth_material,
                 (across, along, self.plinth_height / 2),
                 (self.plinth_width, self.plinth_width, self.plinth_height))
            for number, (across, along) in enumerate(self._places())
        ]

    def busts(self) -> list[Placement]:
        """A bust on the top face of every plinth, looking at the walkway."""
        self._refuse_a_crowded_hall()
        return [
            Placement(self.group_format % (number,),
                      (across, along, self.plinth_height),
                      self._facing(across))
            for number, (across, along) in enumerate(self._places())
        ]

    def lights(self) -> list[Light]:
        """Lamps under the plaster, spread evenly down the hall.

        There are :attr:`lamps` of them and no more, because a forward shading
        pass binds a fixed number of lights -- OpenGLContext's takes eight --
        and the ones past that are not dim, they are absent. Twenty lamps down
        a hall is not a brighter hall: it is the first eight of them lighting
        one end of a dark one.

        They run down the middle rather than over the rows. A hall this narrow
        is crossed by a central lamp, and spending half the budget on each side
        halves how far down the hall the light reaches.
        """
        height = self.height - self.beam_depth - 0.15
        span = self.length / self.lamps
        return [
            Light('Lamp_%03d' % (index,),
                  (0.0, -self.length / 2 + (index + 0.5) * span, height),
                  power=self.lamp_power)
            for index in range(self.lamps)
        ]

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
        subject = next(bust for bust in self.busts()
                       if bust.position[0] == inner)
        across, along, _ = subject.position
        inward = 1.0 if across < 0 else -1.0
        stand = across + inward * 1.1
        return [
            Viewpoint('Gallery', (0.0, self.length / 2 - 2.0, 1.65), turn=0.0),
            # Turned a quarter about to face back across the walkway at it.
            Viewpoint('Bust', (stand, along, 1.45), turn=-inward * math.pi / 2),
        ]

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
