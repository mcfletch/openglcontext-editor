"""A circuit's start/finish gantry, written into the tiles and the tileset.

The frame and its painted line go into the tile as one mesh reading one picture,
so the marker costs a world one draw. The legs also go into the tileset's
``extras`` as props: tile geometry comes and goes with the camera, and a car
that hits a gantry leg has to hit it whatever the streamer is doing.
"""
import io
import json

import numpy as np
import pytest
from OpenGLContext.scenegraph.gantry import GantryProfile, gantry_legs
from OpenGLContext.scenegraph.props import Prop
from OpenGLContext.scenegraph.road import RoadProfile

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.bake.gantry import (
    ATLAS_IMAGE,
    LEG_KIND,
    MAXIMUM_ERROR,
    GantryLayer,
)
from OpenGLContext_editor.world.gantry import start_finish
from OpenGLContext_editor.world.road import RoadPath

ROAD = RoadProfile(lane_width=3.6, lanes=2)
EVERYWHERE = BoundingBox((-500.0, -500.0, -500.0), (500.0, 500.0, 500.0))
ELSEWHERE = BoundingBox((400.0, -50.0, 400.0), (500.0, 50.0, 500.0))


def _path(height=0.0):
    z = np.linspace(0.0, 1000.0, 101)
    return RoadPath(np.stack([np.zeros(101), np.full(101, height), z], axis=-1),
                    profile=ROAD)


def _layer(**kwargs):
    kwargs.setdefault('placement', start_finish(_path()))
    kwargs.setdefault('cell_pixels', 32)
    return GantryLayer(**kwargs)


class TestWhatItPutsInATile:
    def test_the_tile_holding_the_line_gets_the_gantry(self) -> None:
        found = _layer().content(EVERYWHERE, error=1.0)
        assert len(found) == 1
        assert found[0].name == 'gantry'

    def test_a_tile_somewhere_else_gets_nothing(self) -> None:
        assert _layer().content(ELSEWHERE, error=1.0) == []

    def test_a_tile_too_coarse_to_show_it_gets_nothing(self) -> None:
        assert _layer().content(EVERYWHERE, error=MAXIMUM_ERROR + 1.0) == []

    def test_a_gantry_is_visible_from_further_off_than_a_sign(self) -> None:
        """It is nine metres wide and seven tall, and a driver wants to see the
        line coming."""
        from OpenGLContext_editor.bake.signs import MAXIMUM_ERROR as SIGNS
        assert MAXIMUM_ERROR > SIGNS

    def test_the_frame_and_its_line_are_one_mesh(self) -> None:
        mesh = _layer().content(EVERYWHERE, error=1.0)[0].mesh
        assert len(mesh.texcoords) == len(mesh.positions)
        assert len(np.asarray(mesh.indices)) % 3 == 0

    def test_it_is_written_where_the_placement_says(self) -> None:
        placement = start_finish(_path(height=12.0), station=300.0)
        points = np.asarray(
            _layer(placement=placement).content(EVERYWHERE, error=1.0)[0].mesh
            .positions, dtype='d')
        assert pytest.approx(float(placement.position[2]),
                             abs=1.0) == float(points[:, 2].mean())
        assert float(points[:, 1].min()) < 12.1

    def test_the_line_is_painted_at_the_road_surface(self) -> None:
        placement = start_finish(_path(height=12.0))
        points = np.asarray(
            _layer(placement=placement).content(EVERYWHERE, error=1.0)[0].mesh
            .positions, dtype='d')
        near_the_road = points[np.abs(points[:, 1] - 12.0) < 0.5]
        assert len(near_the_road) >= 6
        assert float(np.ptp(near_the_road[:, 0])) >= ROAD.carriageway_width - 1e-3


class TestWhereItReachesTo:
    def test_it_covers_the_whole_frame(self) -> None:
        profile = GantryProfile()
        box = _layer(profile=profile).bounds()
        placement = start_finish(_path())
        assert box.maximum[1] >= placement.position[1] + profile.height - 1e-6
        assert box.maximum[0] - box.minimum[0] >= placement.span

    def test_it_reaches_down_to_the_lowest_foot(self) -> None:
        placement = start_finish(
            _path(), ground=lambda x, z: np.where(np.asarray(x) > 0.0, -6.0, 0.0))
        box = _layer(placement=placement).bounds()
        assert box.minimum[1] <= -6.0


class TestThePictureItReads:
    def test_the_atlas_is_written_once_beside_the_tileset(self) -> None:
        assets = _layer().assets()
        assert list(assets) == [ATLAS_IMAGE]

    def test_it_is_a_readable_image(self) -> None:
        from PIL import Image
        image = Image.open(io.BytesIO(_layer().assets()[ATLAS_IMAGE]))
        assert image.size == (64, 64)

    def test_the_tile_names_the_file_rather_than_carrying_it(self) -> None:
        mesh = _layer().content(EVERYWHERE, error=1.0)[0].mesh
        texture = mesh.material.textures['baseColor']
        assert texture.uri == ATLAS_IMAGE


class TestWhatAGameIsToldAboutIt:
    def test_the_world_says_where_the_lap_begins(self) -> None:
        found = _layer().metadata()['start']
        assert pytest.approx([0.0, 0.0, 0.0], abs=1e-3) == found['at']
        assert pytest.approx(0.0, abs=1e-4) == found['yaw']
        assert pytest.approx(ROAD.carriageway_width, abs=1e-3) == found['width']

    def test_the_legs_are_solid(self) -> None:
        props = _layer().metadata()['props']
        assert len(props) == 2
        assert {one['kind'] for one in props} == {LEG_KIND}

    def test_a_leg_stands_where_it_is_drawn(self) -> None:
        placement = start_finish(_path())
        props = [Prop.from_json(one) for one in _layer().metadata()['props']]
        offsets = sorted(float(one.position[0]) for one in props)
        assert pytest.approx([-placement.span / 2.0, placement.span / 2.0],
                             abs=1e-3) == offsets

    def test_a_leg_is_as_tall_as_the_gantry(self) -> None:
        placement = start_finish(_path())
        legs = gantry_legs(placement.span, GantryProfile(), placement.drops)
        props = [Prop.from_json(one) for one in _layer().metadata()['props']]
        assert pytest.approx(legs[0].height,
                             abs=1e-3) == min(one.height for one in props)

    def test_a_leg_on_low_ground_stands_on_that_ground(self) -> None:
        placement = start_finish(
            _path(), ground=lambda x, z: np.where(np.asarray(x) > 0.0, -6.0, 0.0))
        props = [Prop.from_json(one)
                 for one in _layer(placement=placement).metadata()['props']]
        assert pytest.approx(-6.25, abs=0.01) == min(
            float(one.position[1]) for one in props)

    def test_a_leg_is_no_wider_than_the_post_it_is(self) -> None:
        """A body wider than the geometry is a car stopping short of thin air."""
        props = [Prop.from_json(one) for one in _layer().metadata()['props']]
        assert all(one.radius <= GantryProfile().leg_radius + 1e-6
                   for one in props)


class TestInAWholeWorld:
    def test_it_bakes_alongside_a_landscape(self, tmp_path) -> None:
        from OpenGLContext_editor.bake.driver import bake_world
        from OpenGLContext_editor.bake.layers import HeightfieldLayer

        ground = BoundingBox((-256.0, 0.0, -256.0), (256.0, 0.0, 256.0))
        result = bake_world(
            [HeightfieldLayer(height_fn=lambda x, z: np.zeros_like(np.asarray(x, 'd')),
                              extent=ground, resolution=9),
             _layer()],
            directory=str(tmp_path), depth=2, name='circuit.json')
        document = json.loads((tmp_path / 'circuit.json').read_text())
        assert document['extras']['start']['at'] == [0.0, 0.0, 0.0]
        assert len(document['extras']['props']) == 2
        assert (tmp_path / ATLAS_IMAGE).exists()
        assert result.tiles >= 1


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
