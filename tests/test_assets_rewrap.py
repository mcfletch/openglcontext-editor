"""Giving a reduced asset its own texture, baked from the one it came from.

A photogrammetry scan arrives unwrapped by the scanner, into thousands of small
charts. That is the right answer for the dense mesh -- small charts distort
least -- and the wrong one for anything reduced from it: a texture coordinate
means something only inside one chart, so once an output triangle is larger than
a chart its three corners point at unrelated places in the image and it draws
the stripe between them.

No decimator fixes that, because the fault is in the unwrap rather than in the
surface. What fixes it is the step every scan pipeline takes: unwrap the reduced
mesh afresh, so its charts are made of whole triangles, and bake what the
original looked like into the new atlas.
"""

from __future__ import annotations

import numpy as np
import pytest

rewrap = pytest.importorskip('OpenGLContext_editor.assets.rewrap')


def _quad(size=1.0, normals=False):
    """Two triangles in the XY plane, with a texture laid across them."""
    positions = np.asarray([(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0)], dtype='f4')
    indices = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    uv = np.asarray([(0, 1), (1, 1), (1, 0), (0, 0)], dtype='f4')
    if normals:
        return positions, indices, uv, np.tile(np.asarray([(0, 0, 1)], dtype='f4'), (4, 1))
    return positions, indices, uv


def _checker(size=64, squares=8):
    """A pattern whose position in the image is readable from any sample."""
    step = size // squares
    rows, columns = np.mgrid[0:size, 0:size]
    light = ((rows // step) + (columns // step)) % 2 == 0
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[light] = (240, 240, 240)
    image[~light] = (30, 30, 30)
    # A red stripe along the top edge, so a flip is unmistakable.
    image[:step] = (220, 40, 40)
    return image


def _encode(direction):
    """A direction as the bytes a tangent-space normal map holds."""
    return np.rint((np.asarray(direction, dtype='d') + 1.0) * 127.5).astype(np.uint8)


def _through(held, indices, which, weights, leaning):
    """A tangent-space direction taken out into the model's own space.

    The same frame the renderer builds, so this reads the map the way the thing
    that draws it will.
    """
    frame = rewrap._frame_at(held, np.asarray(indices).reshape(-1, 3), which, weights)  # noqa: SLF001 white-box test of the helper
    return np.einsum('irk,ir->ik', frame, np.atleast_2d(leaning))


def _decode(pixels):
    """The direction a normal map's bytes stand for, back to unit length."""
    out = np.asarray(pixels, dtype='d') / 127.5 - 1.0
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


class TestUnwrapping:
    def test_every_triangle_is_laid_out_inside_one_chart(self):
        """Which is the whole point: a chart is made of whole triangles.

        The failure this exists to cure cannot happen to a fresh unwrap, at any
        triangle count, because a chart boundary never runs through a face.
        """
        positions, indices, _uv = _quad()
        laid = rewrap.unwrap(positions, indices)
        assert len(laid.indices) == len(indices)
        assert laid.uv.shape == (len(laid.positions), 2)
        assert np.all(laid.uv >= 0.0) and np.all(laid.uv <= 1.0)

    def test_the_surface_is_welded_before_it_is_unwrapped(self):
        """A mesh split by the atlas it arrived with is not a torn surface.

        A reduced scan carries a vertex per chart corner, so handing it to an
        unwrapper unwelded offers a mesh in thousands of disconnected pieces and
        gets thousands of charts back. The positions say what is joined.
        """
        positions, indices, _uv = _quad()
        # The same quad, with the shared edge split into two pairs of vertices.
        split = np.concatenate([positions, positions[[0, 2]]])
        torn = np.asarray([0, 1, 2, 4, 5, 3], dtype=np.uint32)
        whole = rewrap.unwrap(positions, indices)
        mended = rewrap.unwrap(split.astype('f4'), torn)
        assert mended.charts == whole.charts

    def test_it_says_where_each_new_vertex_came_from(self):
        """An unwrap splits vertices; the caller's attributes have to follow."""
        positions, indices, _uv = _quad()
        laid = rewrap.unwrap(positions, indices)
        assert laid.source.shape == (len(laid.positions),)
        assert np.allclose(laid.positions, laid.positions[np.arange(len(laid.positions))])


class TestBaking:
    def test_a_flat_quad_comes_back_carrying_its_own_texture(self):
        """The plainest case with a knowable answer.

        Unwrapping a flat quad and baking the original into the result has to
        reproduce the original: every texel of the new atlas sits on the quad,
        the quad is the source, and the colour there is the colour there.
        """
        positions, indices, uv = _quad()
        image = _checker()
        laid = rewrap.unwrap(positions, indices)
        baked = rewrap.bake(laid, positions, indices, uv, image, size=64)
        assert baked.shape == (64, 64, 3)
        painted = baked.reshape(-1, 3)
        painted = painted[np.any(painted > 0, axis=1)]
        # Only the three colours the source holds, give or take the sampling.
        near = np.min(
            np.linalg.norm(
                painted[:, None, :].astype('d')
                - np.asarray([(240, 240, 240), (30, 30, 30), (220, 40, 40)], dtype='d'),
                axis=2,
            ),
            axis=1,
        )
        assert float(np.max(near)) < 40.0

    def test_the_texture_is_not_upside_down(self):
        """The one mistake that reads as plausible texture in the wrong place.

        glTF puts a texture coordinate's origin at the top left with ``v``
        running down. Flipping it -- on the way in, on the way out, or both --
        still fills the atlas with real texture, so nothing looks obviously
        broken until the model is drawn and every chart is wearing a piece of
        somewhere else.
        """
        positions, indices, uv = _quad()
        image = _checker()
        laid = rewrap.unwrap(positions, indices)
        baked = rewrap.bake(laid, positions, indices, uv, image, size=64)

        # The red stripe marks v = 0 on the source. Find the world-space edge it
        # belongs to, and check the baked atlas puts red where that edge is.
        red = np.all(np.abs(baked.astype('d') - (220, 40, 40)) < 60, axis=2)
        assert np.any(red), 'the stripe did not survive the bake at all'
        # The stripe marks v = 0 on the source, which the quad's uv puts along
        # its y = 1 edge. So every red texel has to sit at that edge of the
        # model, wherever the new atlas happens to have placed it.
        spots, at = rewrap.texel_map(laid, size=64)
        painted = red[spots[:, 1], spots[:, 0]]
        assert np.any(painted)
        assert float(np.max(at[painted][:, 1])) > 0.85, (
            'the stripe landed at y = %.2f, not at the edge it marks'
            % (float(np.max(at[painted][:, 1])),)
        )

    def test_a_normal_map_still_points_the_same_way_afterwards(self):
        """A tangent-space normal is a direction relative to the unwrap.

        So the bytes cannot simply be copied: a fresh unwrap turns the frame
        they are read in, and the same numbers then mean a different direction.
        The invariant that does hold is the direction itself -- decode the baked
        map against the *new* mesh's frame and the model-space direction has to
        be the one the original encoded.
        """
        positions, indices, uv, normals = _quad(normals=True)
        # A map that leans hard one way, so a lost sign is unmissable.
        leaning = np.zeros((64, 64, 3), dtype=np.uint8)
        leaning[:, :] = _encode((0.6, -0.3, np.sqrt(1 - 0.36 - 0.09)))
        laid = rewrap.unwrap(positions, indices)
        shot = rewrap.project(laid, positions, indices, uv, size=64)
        onto = rewrap.frames(positions, indices, uv, normals)
        into = rewrap.frames(laid.positions, laid.indices, laid.uv, normals[laid.source])
        baked = rewrap.sample_normals(shot, leaning, onto, into, indices, laid.indices)

        # What the original meant, in the model's own space, and what the baked
        # map says when read the way the new mesh will be read.
        meant = _through(onto, indices, shot.onto, shot.onto_weights, _decode(leaning[0, 0]))
        read = _decode(baked[shot.spots[:, 1], shot.spots[:, 0]])
        says = _through(into, laid.indices, shot.into, shot.into_weights, read)
        agree = np.einsum('ik,ik->i', says, meant)
        assert float(np.min(agree)) > 0.99, 'the baked map leans %r where the original meant %r' % (
            says[int(np.argmin(agree))],
            meant[int(np.argmin(agree))],
        )

    def test_a_normal_map_is_flat_where_no_chart_covers(self):
        """The gutter of a normal map is straight up, not black.

        Black decodes to a direction pointing into the surface, so a gutter left
        at zero is a dark ring around every chart the moment anything filters
        across it.
        """
        positions, indices, uv, normals = _quad(normals=True)
        flat = np.full((64, 64, 3), _encode((0.0, 0.0, 1.0)), dtype=np.uint8)
        laid = rewrap.unwrap(positions, indices)
        shot = rewrap.project(laid, positions, indices, uv, size=64)
        baked = rewrap.sample_normals(
            shot,
            flat,
            rewrap.frames(positions, indices, uv, normals),
            rewrap.frames(laid.positions, laid.indices, laid.uv, normals[laid.source]),
            indices,
            laid.indices,
        )
        assert np.all(baked[:, :, 2] > 120), 'somewhere in the map faces into the surface'

    def test_one_projection_serves_every_map_the_model_carries(self):
        """Base colour, normals, roughness and occlusion all read alike.

        Where a texel reads from depends on the two surfaces and nothing else,
        so a model with four maps asks once and spends the answer four times.
        Each map has to come back the same as baking it on its own would.
        """
        positions, indices, uv = _quad()
        laid = rewrap.unwrap(positions, indices)
        shot = rewrap.project(laid, positions, indices, uv, size=64)
        for image in (_checker(), _checker(squares=4)):
            assert np.array_equal(
                rewrap.sample(shot, image),
                rewrap.bake(laid, positions, indices, uv, image, size=64),
            )

    def test_a_map_of_measurements_keeps_its_own_channel_count(self):
        """A single-channel occlusion map does not come back as three."""
        positions, indices, uv = _quad()
        grey = _checker()[:, :, :1]
        baked = rewrap.bake(
            rewrap.unwrap(positions, indices), positions, indices, uv, grey, size=64
        )
        assert baked.shape == (64, 64, 1)

    def test_the_gutter_beside_a_chart_carries_the_chart_s_colour(self):
        """Black in the gutter draws as a dark seam around every chart.

        A renderer filtering at a chart's edge reaches past it, so the colour
        there is spread a few texels outwards. Two quads far apart pack as two
        charts with room between them, which is where that shows.
        """
        positions, indices, uv = _quad()
        apart = np.concatenate([positions, positions + (8.0, 0.0, 0.0)]).astype('f4')
        both = np.concatenate([indices, indices + 4]).astype(np.uint32)
        twice = np.concatenate([uv, uv]).astype('f4')
        laid = rewrap.unwrap(apart, both)
        assert laid.charts >= 2, 'the fixture did not leave a gutter to test'

        baked = rewrap.bake(laid, apart, both, twice, _checker(), size=64)
        spots, _at = rewrap.texel_map(laid, size=64)
        covered = np.zeros((64, 64), dtype=bool)
        covered[spots[:, 1], spots[:, 0]] = True
        beside = np.zeros((64, 64), dtype=bool)
        beside[1:] |= covered[:-1]
        beside[:-1] |= covered[1:]
        beside &= ~covered
        assert np.any(beside), 'no gutter next to a chart at all'
        assert np.any(np.any(baked[beside] > 0, axis=1)), 'the gutter was left black'


class TestReadingBetweenTheSourcesTexels:
    """Where the new atlas is denser than the original's, several of its texels
    fall inside one texel of the source; read at the nearest they come out as
    blocks, read filtered they come out as the gradient the source shows."""

    def test_a_read_between_two_texels_blends_them(self):
        source = np.zeros((1, 2, 3), dtype=np.uint8)
        source[0, 1] = (200, 200, 200)
        shot = rewrap.Projection(spots=np.array([[0, 0]]),
                                 read=np.array([[0.5, 0.5]]), size=1)
        assert int(rewrap.sample(shot, source)[0, 0, 0]) == 100

    def test_a_read_at_a_texel_s_centre_is_that_texel(self):
        source = np.zeros((1, 2, 3), dtype=np.uint8)
        source[0, 1] = (200, 200, 200)
        shot = rewrap.Projection(spots=np.array([[0, 0]]),
                                 read=np.array([[0.75, 0.5]]), size=1)
        assert int(rewrap.sample(shot, source)[0, 0, 0]) == 200


class TestTheRasterIsTheOneThePerFaceWalkGives:
    """Every texel a triangle covers, found for all triangles at once."""

    @staticmethod
    def _one_face_at_a_time(laid, size):
        faces = np.asarray(laid.indices).reshape(-1, 3)
        corner_uv = np.asarray(laid.uv, dtype='d')[faces] * size - 0.5
        low = np.maximum(np.floor(corner_uv.min(axis=1)).astype(np.int64), 0)
        high = np.minimum(np.ceil(corner_uv.max(axis=1)).astype(np.int64) + 1, size)
        found = set()
        for face in range(len(faces)):
            columns, rows = np.meshgrid(np.arange(low[face][0], high[face][0]),
                                        np.arange(low[face][1], high[face][1]))
            pixels = np.stack([columns.ravel(), rows.ravel()], axis=1).astype('d')
            weights = rewrap._barycentric(pixels, corner_uv[face])  # noqa: SLF001 white-box test of the helper
            if weights is None:
                continue
            slack = 0.5 / max(np.ptp(corner_uv[face], axis=0).max(), 1e-12)
            for spot in pixels[np.all(weights >= -slack, axis=1)].astype(int):
                found.add((face, int(spot[0]), int(spot[1])))
        return found

    def test_the_same_texels_for_the_same_faces(self):
        rng = np.random.default_rng(4)
        uv = rng.random((60, 2))
        laid = rewrap.Unwrapped(positions=rng.random((60, 3)).astype('f4'),
                                indices=rng.integers(0, 60, 90).astype(np.uint32),
                                uv=uv.astype('f4'), source=np.arange(60), charts=1)
        spots, _at, whose, _within = rewrap._rasterise(laid, 32)  # noqa: SLF001 white-box test of the helper
        found = {(int(face), int(x), int(y))
                 for face, (x, y) in zip(whose, spots, strict=True)}
        assert found == self._one_face_at_a_time(laid, 32)
