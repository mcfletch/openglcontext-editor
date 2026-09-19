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
    def test_it_has_a_floor_walls_and_a_ceiling(self, plan):
        named = {slab.name for slab in plan.room()}

        assert named == {'Floor', 'Wall_North', 'Wall_South', 'Wall_East',
                         'Wall_West', 'Ceiling'}

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


class TestTheCeiling:
    def test_the_beams_cross_the_hall(self, plan):
        beam = plan.beams()[0]

        assert beam.size[0] >= plan.width

    def test_they_are_the_dark_wood(self, plan):
        assert {beam.material for beam in plan.beams()} == {'GalleryBeam'}

    def test_there_is_one_per_bay(self, plan):
        assert len(plan.beams()) == plan.bays

    def test_they_hang_below_the_plaster(self, plan):
        ceiling = next(slab for slab in plan.room() if slab.name == 'Ceiling')
        top_of_beam = max(beam.centre[2] + beam.size[2] / 2 for beam in plan.beams())

        assert top_of_beam == pytest.approx(ceiling.centre[2] - ceiling.size[2] / 2)

    def test_they_are_clear_of_a_walking_head(self, plan):
        lowest = min(beam.centre[2] - beam.size[2] / 2 for beam in plan.beams())

        assert lowest > 2.0


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
            assert bust.position[:2] == plinth.centre[:2]

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

    def test_the_viewpoints_are_named_apart(self, plan):
        names = [view.name for view in plan.cameras()]

        assert len(set(names)) == len(names)

    def test_there_is_light(self, plan):
        assert plan.lights()

    def test_the_lights_are_under_the_ceiling(self, plan):
        for light in plan.lights():
            assert light.position[2] < plan.height

    def test_there_are_no_more_lamps_than_a_pass_will_bind(self, plan):
        """A ninth lamp is not a dimmer hall, it is a lamp never switched on."""
        assert len(plan.lights()) <= 8

    def test_the_lamps_are_spread_the_length_of_the_hall(self, plan):
        """All of them at one end lights one end and leaves the rest dark."""
        along = sorted(light.position[1] for light in plan.lights())

        assert along[0] < -plan.length / 4
        assert along[-1] > plan.length / 4

    def test_they_are_evenly_spaced(self, plan):
        along = sorted(light.position[1] for light in plan.lights())
        gaps = [second - first
                for first, second in zip(along, along[1:], strict=False)]

        assert max(gaps) == pytest.approx(min(gaps))


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
