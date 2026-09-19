"""The add-on's fold and the engine's fold are the same fold.

An octahedral impostor is a bargain between two programs: the baker renders one
picture per direction and lays them out on a square, and the renderer works out
which one to read from where the viewer is standing. They agree or the impostor
shows the wrong face from every angle, and it is not a crash -- it is a model
that looks subtly, unaccountably wrong.

The arithmetic therefore exists three times: here in the add-on, which installs
into Blender carrying nothing of OpenGLContext; in the engine, which has to read
it back; and in ``pbr.vert``, which has to run on a GPU. The engine's own GL
test holds the shader to the engine's Python. This holds the add-on's Python to
the engine's, which is the seam this project sits on -- it is the one place both
are importable at once.
"""
import numpy as np
import pytest
from OpenGLContext.scenegraph import octahedral as engine

from OpenGLContext_editor.blender.openglcontext_lod import octahedral as addon


def directions(count, seed, upper_only):
    rng = np.random.default_rng(seed)
    made = rng.normal(size=(count, 3))
    made /= np.linalg.norm(made, axis=1)[:, None]
    if upper_only:
        made[:, 1] = np.abs(made[:, 1])
    return made


class TestTheyFoldTheSameWay:
    @pytest.mark.parametrize('hemi', [True, False])
    def test_every_direction_lands_in_the_same_place(self, hemi):
        for direction in directions(2000, seed=19, upper_only=hemi):
            assert addon.direction_to_uv(direction, hemi) == \
                pytest.approx(engine.direction_to_uv(direction, hemi), abs=1e-12)

    @pytest.mark.parametrize('hemi', [True, False])
    def test_every_place_unfolds_to_the_same_direction(self, hemi):
        rng = np.random.default_rng(23)
        for uv in rng.random((2000, 2)):
            assert addon.uv_to_direction(uv, hemi) == \
                pytest.approx(engine.uv_to_direction(uv, hemi), abs=1e-12)

    @pytest.mark.parametrize('grid', [2, 3, 8, 12])
    def test_they_hold_the_same_views(self, grid):
        assert addon.view_directions(grid) == \
            pytest.approx(engine.view_directions(grid))

    @pytest.mark.parametrize('grid', [4, 8])
    def test_they_choose_the_same_tile(self, grid):
        for direction in directions(1000, seed=29, upper_only=True):
            assert addon.cell_of(direction, grid) == \
                engine.cell_of(direction, grid)


class TestTheySayTheSameAboutAnAtlas:
    def test_the_same_tile_size(self):
        assert addon.tile_size(256, 8) == engine.tile_size(256, 8)

    def test_the_same_tile_origins(self):
        for row in range(4):
            for column in range(4):
                assert addon.tile_origin(row, column, 128, 4) == \
                    engine.tile_origin(row, column, 128, 4)

    def test_the_same_floor_on_how_small_a_view_may_be(self):
        assert addon.MINIMUM_TILE == engine.MINIMUM_TILE
