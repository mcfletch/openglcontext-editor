"""Where the gallery puts its walls, its plinths and its busts.

The demo world is a room and a grid, and both are arithmetic: a hall of a given
size, plinths down it in rows, a bust centred on each, beams across the ceiling
between the plaster. None of that needs Blender to work out, so none of it is
worked out inside Blender -- the add-on asks this module where things go and
then makes them.

The measurements matter to the demo rather than only to the picture. Every
level of a chain has to be on screen at once for the switching to be watchable,
and that is a fact about how long the hall is and how big a bust is.
"""

import math

import pytest

from OpenGLContext_editor.blender.openglcontext_lod import gallery


@pytest.fixture
def plan():
    return gallery.Gallery()


class TestTheRoom:
    def test_it_has_a_floor_and_four_walls(self, plan):
        """And no roof: the sky is what is overhead."""
        named = {slab.name for slab in plan.room()}

        assert named == {'Floor', 'Wall_North', 'Wall_South', 'Wall_East',
                         'Wall_West'}

    def test_the_floor_is_the_whole_hall(self, plan):
        floor = next(slab for slab in plan.room() if slab.name == 'Floor')

        assert floor.size[0] == pytest.approx(plan.width)
        assert floor.size[1] == pytest.approx(plan.length)

    def test_the_floor_is_the_polished_wood(self, plan):
        floor = next(slab for slab in plan.room() if slab.name == 'Floor')

        assert floor.material == 'GalleryFloor'

    def test_the_walls_are_plaster(self, plan):
        walls = [slab for slab in plan.room() if slab.name.startswith('Wall')]

        assert {slab.material for slab in walls} == {'GalleryWall'}

    def test_the_floor_sits_below_the_room(self, plan):
        """Its top face is z=0, so everything else can stand on zero."""
        floor = next(slab for slab in plan.room() if slab.name == 'Floor')

        assert floor.centre[2] + floor.size[2] / 2 == pytest.approx(0.0)

    def test_the_side_walls_run_the_length_of_the_hall(self, plan):
        east = next(slab for slab in plan.room() if slab.name == 'Wall_East')

        assert east.size[1] >= plan.length


class TestTheBeams:
    def test_the_beams_cross_the_hall(self, plan):
        beam = plan.beams()[0]

        assert beam.size[0] >= plan.width

    def test_they_are_the_dark_wood(self, plan):
        assert {beam.material for beam in plan.beams()} == {'GalleryBeam'}

    def test_there_is_one_per_bay(self, plan):
        assert len(plan.beams()) == plan.bays

    def test_they_span_the_tops_of_the_walls(self, plan):
        """Where the roof would have sat: the sun comes through between them,
        which is what puts the stripes on the busts."""
        top_of_beam = max(beam.centre[2] + beam.size[2] / 2 for beam in plan.beams())

        assert top_of_beam == pytest.approx(plan.height)

    def test_the_sky_shows_between_them(self, plan):
        """A bay is wider than a beam, or the hall is roofed in timber."""
        assert plan.beam_width < plan.bay

    def test_they_are_clear_of_a_walking_head(self, plan):
        lowest = min(beam.centre[2] - beam.size[2] / 2 for beam in plan.beams())

        assert lowest > 2.0


class TestHowTheyStand:
    """Nothing in a room is exactly where it was meant to be.

    A hundred and twenty plinths on a perfect lattice, each square to the hall
    and each bust facing exactly the same way, reads as a rendering rather than
    as a room -- the eye finds the repetition before it finds the busts. Each
    one is moved by a couple of millimetres and turned by a fraction of a
    degree, which is what a floor and a pair of hands do to a stone.
    """

    def test_they_are_not_all_in_the_same_place_in_their_bay(self, plan):
        offsets = {(round(plinth.centre[0], 6), round(plinth.centre[1], 6))
                   for plinth in plan.plinths()}

        assert len(offsets) == len(plan.plinths())

    def test_none_of_them_moves_more_than_a_few_millimetres(self, plan):
        """It is a hall that has been walked through, not an earthquake."""
        for plinth, place in zip(plan.plinths(), plan._places(), strict=True):
            assert math.dist(plinth.centre[:2], place) <= plan.plinth_shift

    def test_they_are_turned_a_fraction_of_a_degree(self, plan):
        turns = [plinth.turn for plinth in plan.plinths()]

        assert max(abs(turn) for turn in turns) <= plan.plinth_turn
        assert len(set(turns)) > len(turns) // 2

    def test_each_bust_stays_on_its_plinth(self, plan):
        """A bust carries its plinth's own shift and adds a smaller one: what
        it must not do is wander off the stone it stands on."""
        for bust, plinth in zip(plan.busts(), plan.plinths(), strict=True):
            assert math.dist(bust.position[:2], plinth.centre[:2]) <= plan.bust_shift

    def test_the_shift_is_small_beside_the_stone_it_stands_on(self, plan):
        """Millimetres against a plinth two thirds of a metre across."""
        assert plan.bust_shift < plan.plinth_width / 20.0
        assert plan.plinth_shift < plan.plinth_width / 20.0

    def test_the_busts_are_not_all_facing_the_same_way(self, plan):
        left = [bust.turn for bust in plan.busts() if bust.position[0] < 0]

        assert len(set(left)) > len(left) // 2

    def test_no_bust_is_turned_far_enough_to_look_at_a_wall(self, plan):
        for bust in plan.busts():
            square = plan._facing(bust.position[0])
            assert abs(bust.turn - square) <= plan.bust_turn

    def test_the_same_hall_is_built_twice_the_same(self, plan):
        """A world rebuilt is the same world, or every capture of it differs."""
        again = gallery.Gallery()

        assert [plinth.centre for plinth in again.plinths()] == \
            [plinth.centre for plinth in plan.plinths()]
        assert [bust.turn for bust in again.busts()] == \
            [bust.turn for bust in plan.busts()]

    def test_another_seed_stands_them_differently(self):
        one = gallery.Gallery()
        other = gallery.Gallery(jitter_seed=one.jitter_seed + 1)

        assert [p.centre for p in other.plinths()] != [p.centre for p in one.plinths()]


class TestThePlinths:
    def test_one_per_bust(self, plan):
        assert len(plan.plinths()) == len(plan.busts())

    def test_they_are_rectangular_and_upright(self, plan):
        plinth = plan.plinths()[0]

        assert plinth.size[0] == pytest.approx(plinth.size[1])
        assert plinth.size[2] > plinth.size[0]

    def test_they_stand_on_the_floor(self, plan):
        for plinth in plan.plinths():
            assert plinth.centre[2] - plinth.size[2] / 2 == pytest.approx(0.0)

    def test_they_are_inside_the_walls(self, plan):
        for plinth in plan.plinths():
            assert abs(plinth.centre[0]) + plinth.size[0] / 2 < plan.width / 2
            assert abs(plinth.centre[1]) + plinth.size[1] / 2 < plan.length / 2

    def test_no_two_of_them_overlap(self, plan):
        places = [plinth.centre[:2] for plinth in plan.plinths()]

        assert len(set(places)) == len(places)

    def test_a_walkway_is_left_down_the_middle(self, plan):
        """The camera walks the hall; it must not have to walk through stone."""
        for plinth in plan.plinths():
            assert abs(plinth.centre[0]) - plinth.size[0] / 2 > 0.6


class TestTheBusts:
    def test_there_are_hundreds(self, plan):
        assert len(plan.busts()) >= 100

    def test_rows_times_bays(self, plan):
        assert len(plan.busts()) == len(plan.rows) * plan.bays

    def test_each_stands_on_the_top_of_its_plinth(self, plan):
        for bust, plinth in zip(plan.busts(), plan.plinths(), strict=True):
            assert bust.position[2] == pytest.approx(plinth.size[2])
            assert math.dist(bust.position[:2], plinth.centre[:2]) < 0.01

    def test_each_chain_has_a_name_of_its_own(self, plan):
        """One chain per bust: a level belongs to the copy it is a level of."""
        assert len({bust.group for bust in plan.busts()}) == len(plan.busts())

    def test_they_face_the_walkway(self, plan):
        """A bust on the left of the hall and one on the right do not face the
        same way, or half of them are looking at a wall."""
        left = [bust for bust in plan.busts() if bust.position[0] < 0]
        right = [bust for bust in plan.busts() if bust.position[0] > 0]

        assert {bust.turn for bust in left} != {bust.turn for bust in right}


class TestWhatTheHallIsFor:
    def test_the_far_end_is_far_enough_for_the_coarsest_level(self, plan):
        """A chain proves nothing if the hall is too short to reach its end."""
        coverage = gallery.coverage_at(plan.bust_radius, plan.length)

        assert coverage < 0.03

    def test_the_near_end_is_close_enough_for_the_finest(self, plan):
        coverage = gallery.coverage_at(plan.bust_radius, 1.0)

        assert coverage > 0.5

    def test_the_camera_starts_inside_the_hall_looking_down_it(self, plan):
        camera = plan.cameras()[0]

        assert abs(camera.position[1]) < plan.length / 2
        assert 1.0 < camera.position[2] < 2.2

    def test_there_is_a_second_viewpoint_beside_a_bust(self, plan):
        """Walking between the two is the demo; a viewer can also jump."""
        near = plan.cameras()[1]
        closest = min(math.dist(near.position[:2], bust.position[:2])
                      for bust in plan.busts())

        assert closest < 2.0

    def test_the_walk_between_the_viewpoints_stays_in_the_walkway(self, plan):
        """A camera moving from one to the other travels the hall; it must do
        it down the middle rather than through a row of plinths."""
        first, second = plan.cameras()[0], plan.cameras()[1]
        half = plan.plinth_width / 2
        occupied = [(across - half, across + half) for across in plan.rows]

        for step in range(21):
            across = (first.position[0]
                      + (second.position[0] - first.position[0]) * step / 20)
            assert not any(low < across < high for low, high in occupied), across

    def test_every_viewpoint_is_inside_the_room(self, plan):
        for view in plan.cameras():
            assert abs(view.position[0]) < plan.width / 2
            assert abs(view.position[1]) < plan.length / 2
            assert 0.0 < view.position[2] < plan.height

    def test_there_is_a_viewpoint_at_a_bust_s_shoulder(self, plan):
        """One still with the whole demo in it: a bust at arm's length drawing
        at its finest, and the length of the hall behind it drawing at every
        level down to its coarsest."""
        shoulder = next(view for view in plan.cameras()
                        if view.name == 'Shoulder')
        nearest = min(math.dist(shoulder.position[:2], bust.position[:2])
                      for bust in plan.busts())

        assert nearest < 1.0

    def test_the_shoulder_view_is_where_a_viewer_reads_it(self, plan):
        """Stated in the file's own Y-up: (-2.07, 1.65, -22.41)."""
        shoulder = next(view for view in plan.cameras()
                        if view.name == 'Shoulder')
        across, along, up = shoulder.position

        assert (round(across, 2), round(up, 2), round(-along, 2)) == \
            (-2.07, 1.65, -22.41)

    def test_the_shoulder_view_faces_the_far_corner(self, plan):
        shoulder = next(view for view in plan.cameras()
                        if view.name == 'Shoulder')
        looking = (math.sin(shoulder.turn), -math.cos(shoulder.turn))
        corner = plan.far_corner
        wanted = (corner[0] - shoulder.position[0],
                  corner[1] - shoulder.position[1])
        length = math.hypot(*wanted)

        assert looking[0] == pytest.approx(wanted[0] / length, abs=0.01)
        assert looking[1] == pytest.approx(wanted[1] / length, abs=0.01)

    def test_the_viewpoints_are_named_apart(self, plan):
        names = [view.name for view in plan.cameras()]

        assert len(set(names)) == len(names)

    def test_there_is_light(self, plan):
        assert plan.lights()

    def test_the_sun_leans_far_enough_to_throw_a_shadow(self, plan):
        """Straight down is a hall of busts standing on their own shadows."""
        for sun in plan.lights():
            across = math.hypot(sun.direction[0], sun.direction[1])
            assert across > 0.2

    def test_every_light_is_a_unit_direction(self, plan):
        for sun in plan.lights():
            assert math.dist(sun.direction, (0.0, 0.0, 0.0)) == pytest.approx(1.0)

    def test_the_hall_is_lit_for_a_neutral_exposure(self, plan):
        """Every light in the hall counts towards the illuminance the engine's
        light meter reads, and the meter reads six lux as neutral.

        Over that, the hall is exposed correctly only where the meter is
        consulted -- against a black background and nowhere else -- and
        ``oglc-view``'s own default draws a sky. A hall that stays under it
        looks the same either way.
        """
        assert sum(sun.lux for sun in plan.lights()) <= 6.0

    def test_the_lights_are_named_apart(self, plan):
        names = [sun.name for sun in plan.lights()]

        assert len(set(names)) == len(names)


class TestWhatStandsBetweenTheLightAndTheRoom:
    """The shell is lit and takes shadows; it does not throw any.

    Two suns overhead are outside the building. A wall or a ceiling that cast
    would put the whole interior in its shadow, which is a dark hall rather
    than a lit one -- so the shell is marked as not casting, and everything
    standing in the room still is.
    """

    @pytest.fixture
    def plan(self):
        return gallery.Gallery()

    def test_the_walls_cast(self, plan):
        """A rafter's shadow that crosses a wall and does not stop where the
        wall's own shadow begins reads as a shadow floating in the air."""
        walls = [slab for slab in plan.room() if slab.name.startswith('Wall')]

        assert walls and all(slab.casts for slab in walls)

    def test_the_floor_casts_nothing(self, plan):
        """It is the ground: there is nothing under it to shadow, and a floor
        in the depth pass shadows itself along every grazing ray."""
        floor = next(slab for slab in plan.room() if slab.name == 'Floor')

        assert floor.casts is False

    def test_what_stands_in_the_room_casts(self, plan):
        assert all(slab.casts for slab in plan.plinths())
        assert all(slab.casts for slab in plan.beams())

    def test_a_slab_casts_unless_it_is_told_not_to(self):
        assert gallery.Slab('X', 'M', (0, 0, 0), (1, 1, 1)).casts


class TestAskingForADifferentHall:
    def test_fewer_bays_is_fewer_busts(self):
        assert len(gallery.Gallery(bays=4).busts()) == 4 * len(gallery.Gallery().rows)

    def test_a_smaller_hall_still_closes(self):
        small = gallery.Gallery(bays=2, length=6.0)
        floor = next(slab for slab in small.room() if slab.name == 'Floor')

        assert floor.size[1] == pytest.approx(6.0)

    def test_a_hall_too_short_for_its_bays_is_refused(self):
        with pytest.raises(ValueError):
            gallery.Gallery(bays=40, length=4.0).plinths()
