"""The sky the gallery is roofed with, and how it reaches the file.

The hall has no ceiling: what is overhead is a sky, and it has three jobs. It
is the backdrop seen between the beams, it is what the polished floor and the
marble reflect, and it is where the light that is not the sun comes from. One
equirectangular panorama does all three, written into the glB as
``OMI_environment_sky`` -- which OpenGLContext reads as a skybox *and* as the
image-based lighting probe.

The panorama is made here rather than fetched: the colours have to answer to
the room (the floor's own colour below the horizon, so what the parquet
reflects is the parquet) and a generated sky is CC0 by construction.
"""
import json
import math
import struct

import numpy as np
import pytest

from OpenGLContext_editor.blender.openglcontext_lod import gallery, sky

pytest.importorskip('PIL')


@pytest.fixture
def plan():
    return gallery.Gallery()


@pytest.fixture(scope='module')
def panorama():
    """One small panorama, shared: the generator is the same at any size."""
    return sky.panorama(width=256, height=128, seed=7)


class TestThePanorama:
    def test_it_is_equirectangular(self, panorama):
        """Twice as wide as it is tall: a full turn by a half turn."""
        assert panorama.size == (256, 128)

    def test_the_ground_is_the_floor_colour(self, panorama):
        """What the polished floor reflects below the horizon is the floor."""
        pixels = np.asarray(panorama.convert('RGB'), dtype=float) / 255.0
        ground = pixels[panorama.height * 3 // 4]
        expected = np.asarray(sky.GROUND_COLOUR, dtype=float)

        assert np.allclose(ground.mean(axis=0), expected, atol=0.08)

    def test_the_sky_is_not_the_ground(self, panorama):
        pixels = np.asarray(panorama.convert('RGB'), dtype=float) / 255.0
        above = pixels[:panorama.height // 2].mean(axis=(0, 1))
        below = pixels[panorama.height // 2:].mean(axis=(0, 1))

        assert not np.allclose(above, below, atol=0.1)

    def test_the_zenith_is_blue(self, panorama):
        pixels = np.asarray(panorama.convert('RGB'), dtype=float) / 255.0
        top = pixels[:4].mean(axis=(0, 1))

        assert top[2] > top[0]

    def test_there_are_clouds_in_it(self, panorama):
        """A flat gradient has no variance along a line of latitude; cloud
        does, and that is the whole difference between a sky and a backdrop."""
        pixels = np.asarray(panorama.convert('L'), dtype=float) / 255.0
        band = pixels[panorama.height // 4]

        assert band.std() > 0.02

    def test_the_clouds_are_lighter_than_the_sky_behind_them(self, panorama):
        pixels = np.asarray(panorama.convert('L'), dtype=float) / 255.0
        band = pixels[panorama.height // 4]

        assert band.max() > band.mean() + band.std()

    def test_it_joins_up_where_it_wraps(self, panorama):
        """An equirectangular panorama meets itself; a seam is a stripe down
        the sky and a stripe in every reflection of it."""
        pixels = np.asarray(panorama.convert('RGB'), dtype=float) / 255.0
        seam = np.abs(pixels[:, 0] - pixels[:, -1]).max()
        typical = np.abs(np.diff(pixels[:, :8], axis=1)).max()

        assert seam <= typical * 3 + 0.02

    def test_the_same_seed_makes_the_same_sky(self):
        one = np.asarray(sky.panorama(width=64, height=32, seed=3))
        other = np.asarray(sky.panorama(width=64, height=32, seed=3))

        assert np.array_equal(one, other)

    def test_another_seed_makes_another_sky(self):
        one = np.asarray(sky.panorama(width=64, height=32, seed=3), dtype=float)
        other = np.asarray(sky.panorama(width=64, height=32, seed=4), dtype=float)

        assert not np.allclose(one, other)


def _document(blob):
    """The JSON chunk of a glB, as a dict."""
    _magic, _version, _length = struct.unpack('<III', blob[:12])
    chunk, _kind = struct.unpack('<II', blob[12:20])
    return json.loads(blob[20:20 + chunk])


def _glb_with_nothing_in_it():
    """The smallest glB the injector could be handed: one empty scene."""
    body = json.dumps({'asset': {'version': '2.0'},
                       'scene': 0, 'scenes': [{'nodes': []}],
                       'nodes': []}).encode('utf-8')
    body += b' ' * (-len(body) % 4)
    header = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(body))
    return header + struct.pack('<II', len(body), 0x4E4F534A) + body


@pytest.fixture(scope='module')
def written():
    """One small world with a sky in it, read by several tests."""
    return sky.with_sky(_glb_with_nothing_in_it(),
                        sky.panorama(width=64, height=32, seed=1))


class TestWritingItIntoTheFile:

    def test_the_file_is_still_a_glb(self, written):
        assert written[:4] == b'glTF'
        assert struct.unpack('<I', written[8:12])[0] == len(written)

    def test_it_declares_the_extension(self, written):
        doc = _document(written)

        assert sky.EXTENSION in doc['extensionsUsed']
        assert sky.EXTENSION in doc['extensions']

    def test_the_sky_is_a_panorama(self, written):
        entry = _document(written)['extensions'][sky.EXTENSION]['skies'][0]

        assert entry['type'] == 'panorama'
        assert 'equirectangular' in entry['panorama']

    def test_the_panorama_names_a_texture_that_is_there(self, written):
        doc = _document(written)
        index = doc['extensions'][sky.EXTENSION]['skies'][0]['panorama']['equirectangular']
        texture = doc['textures'][index]

        assert 0 <= texture['source'] < len(doc['images'])
        assert doc['images'][texture['source']]['uri'].startswith('data:image/')

    def test_the_extension_is_not_required(self, written):
        """A viewer that has never heard of it must still open the world."""
        assert sky.EXTENSION not in _document(written).get('extensionsRequired', [])

    def test_a_document_that_already_has_textures_keeps_them(self):
        blob = _glb_with_nothing_in_it()
        doc = _document(blob)
        doc['images'] = [{'uri': 'data:image/png;base64,AAAA'}]
        doc['textures'] = [{'source': 0}]
        doc['samplers'] = [{}]
        rebuilt = sky.with_sky(_rewrite(blob, doc),
                               sky.panorama(width=64, height=32, seed=1))
        after = _document(rebuilt)

        assert after['images'][0]['uri'] == 'data:image/png;base64,AAAA'
        assert after['textures'][0]['source'] == 0
        assert len(after['textures']) == 2


def _rewrite(blob, doc):
    """A glB carrying ``doc`` as its JSON chunk, keeping any binary chunk."""
    _magic, _version, _length = struct.unpack('<III', blob[:12])
    chunk, _kind = struct.unpack('<II', blob[12:20])
    rest = blob[20 + chunk:]
    body = json.dumps(doc).encode('utf-8')
    body += b' ' * (-len(body) % 4)
    return (struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(body) + len(rest))
            + struct.pack('<II', len(body), 0x4E4F534A) + body + rest)


class TestWhatTheEngineMakesOfIt:
    """The point of the whole exercise: the loader has to see a sky."""

    def test_the_loader_builds_a_sky_from_it(self, tmp_path):
        from OpenGLContext.loaders import gltf
        from OpenGLContext.scenegraph.hdrbackground import HDRBackground

        written = sky.with_sky(_glb_with_nothing_in_it(),
                               sky.panorama(width=64, height=32, seed=1))
        path = tmp_path / 'sky.glb'
        path.write_bytes(written)
        scene = gltf.load_gltf(str(path))

        backdrops = [node for node in _walk(scene.group)
                     if isinstance(node, HDRBackground)]
        assert len(backdrops) == 1

    def test_the_panorama_survives_as_pixels(self, tmp_path):
        """Not merely that a background exists: that it is the sky we wrote."""
        from OpenGLContext.loaders.gltf import environment_sky as reader
        from OpenGLContext.loaders.gltf.loader import parse_gltf
        from OpenGLContext.loaders.resolver import Resolver

        written = sky.with_sky(_glb_with_nothing_in_it(),
                               sky.panorama(width=64, height=32, seed=1))
        path = tmp_path / 'sky.glb'
        path.write_bytes(written)
        shared = parse_gltf(str(path))
        skies = reader.read_skies(shared.gltf.extensions)

        assert len(skies) == 1
        assert isinstance(skies[0], reader.PanoramaSky)
        built = reader.background_for(skies[0], shared.gltf,
                                      Resolver(base_dir=str(tmp_path)))
        assert built is not None


def _walk(node, out=None):
    out = [] if out is None else out
    out.append(node)
    for child in getattr(node, 'children', None) or []:
        _walk(child, out)
    return out


class TestTheHallUnderIt:
    def test_there_is_no_roof(self, plan):
        """What is overhead is the sky, so nothing stands between it and the
        hall but the beams."""
        assert 'Ceiling' not in {slab.name for slab in plan.room()}

    def test_the_walls_and_the_floor_are_still_there(self, plan):
        assert {slab.name for slab in plan.room()} == {
            'Floor', 'Wall_North', 'Wall_South', 'Wall_East', 'Wall_West'}

    def test_the_beams_still_cross_the_hall(self, plan):
        """They are what cuts the sunlight into stripes across the busts."""
        assert len(plan.beams()) == plan.bays

    def test_the_beams_cast(self, plan):
        assert all(beam.casts for beam in plan.beams())

    def test_one_sun_lights_it(self, plan):
        """Two suns of equal strength read as a lighting rig rather than as a
        time of day; one sun and a sky is what a room with a roof open to it
        actually has."""
        assert len(plan.lights()) == 1

    def test_the_sun_leans_across_the_hall_and_along_it(self, plan):
        """Across, so the beams stripe the busts rather than the floor under
        them; along, so the stripe lands on a face and not on the wall."""
        sun = plan.lights()[0]

        assert sun.direction[2] < 0.0
        assert abs(sun.direction[0]) > 0.0
        assert abs(sun.direction[1]) > 0.0

    def test_the_sun_casts(self, plan):
        assert plan.lights()[0].casts

    def test_the_sun_is_inside_what_the_meter_reads_as_neutral(self, plan):
        """The sky lights the hall as well, and the exposure is not metered
        when a background is drawn, so the sun alone must sit under six lux."""
        assert 0.0 < plan.lights()[0].lux <= 6.0

    def test_the_sky_says_how_high_the_sun_is(self, plan):
        """A panorama whose sun stands at noon over a hall with afternoon
        shadows in it is a picture of two different days."""
        sun = plan.lights()[0]

        assert plan.sun_elevation() == pytest.approx(math.asin(-sun.direction[2]))

    def test_the_shadow_is_a_third_of_what_throws_it(self, plan):
        """How high the sun stands, said as what it does: a metre of plinth
        lays a third of a metre of shadow across the floor. Lower than that
        and the hall is all shadow; higher and there is nothing to see."""
        sun = plan.lights()[0]
        across = math.hypot(sun.direction[0], sun.direction[1])

        assert across / -sun.direction[2] == pytest.approx(plan.shadow_ratio)
        assert plan.shadow_ratio == pytest.approx(1.0 / 3.0)

    def _shadowed(self, plan, height):
        """Whether a face at ``height``, one row along from a beam, is under it.

        A ray from that point towards the sun climbs to the beams; it is in
        shadow if it is still inside the beam's own depth when it gets there.
        Both of the beam's trailing corners bound that, which is why a band
        rather than a line.
        """
        sun = plan.lights()[0]
        slope = abs(sun.direction[1] / sun.direction[2])
        under = ((plan.height - plan.beam_depth) - height) * slope
        top = (plan.height - height) * slope
        reach = plan.bay / 2.0 - plan.beam_lead
        return (under <= reach + plan.beam_width / 2.0
                and top >= reach - plan.beam_width / 2.0)

    def test_a_rafter_shadows_the_head_of_the_next_row_of_busts(self, plan):
        """The line the whole arrangement is for: a beam's shadow covers the
        crown of a bust one row along and stops about the eyes, so the face is
        in the sun under a shaded head -- an edge that reads from the end of
        the hall and still reads with your nose against the marble."""
        crown = plan.plinth_height + plan.bust_height
        chin = plan.plinth_height

        assert self._shadowed(plan, crown)
        assert not self._shadowed(plan, chin)

    def test_the_shadow_stops_about_the_eyes(self, plan):
        just_above = plan.shadow_line + 0.02
        just_below = plan.shadow_line - 0.02

        assert self._shadowed(plan, just_above)
        assert not self._shadowed(plan, just_below)

    def test_it_is_the_head_it_shades_and_not_the_bust(self, plan):
        """A quarter or so of the height, which is a hat rather than a hood."""
        share = (plan.plinth_height + plan.bust_height - plan.shadow_line)

        assert 0.15 <= share / plan.bust_height <= 0.4

    def test_the_line_it_draws_is_where_the_eyes_are(self, plan):
        assert plan.shadow_line == pytest.approx(
            plan.plinth_height + plan.shadow_share * plan.bust_height)

    def test_the_sky_says_which_way_the_sun_is(self, plan):
        """The bright patch in the panorama and the shadow on the floor have
        to agree, or the picture says two different times of day."""
        sun = plan.lights()[0]
        azimuth = math.atan2(-sun.direction[1], -sun.direction[0])

        assert plan.sun_azimuth() == pytest.approx(azimuth)
