"""Placing a forest without generating four of them first.

Uniform random candidates thinned to a minimum spacing is dart-throwing: to
saturate the packing it has to be given several times the number of instances it
will keep, and every one of those costs a height lookup, four more for the slope
and a distance-to-road query. On a four-kilometre world that is eight million
candidates for half a million trees, and it is most of what a bake spends.

A *jittered grid* at the spacing asks for the answer directly. Every cell holds
one candidate, no two candidates in neighbouring cells can be closer than the
jitter allows, and the thinning that follows has a set that is already nearly
right rather than one that is four times too dense.

The filters then run cheapest-first on what is left of the set, rather than all
of them on all of it: the elevation band is free once the height is known, and
the slope costs four more height lookups apiece.
"""

import math

import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.scatter import scatter_on_heightfield

SIDE = 400.0
REGION = BoundingBox((-SIDE / 2, 0.0, -SIDE / 2), (SIDE / 2, 0.0, SIDE / 2))


def _flat(x, z):
    return np.zeros(np.shape(np.asarray(x, dtype='d')))


class _Counting:
    """A height function that says how many points it was asked about."""

    def __init__(self, inner=_flat):
        self.inner = inner
        self.points = 0

    def __call__(self, x, z):
        found = np.asarray(x, dtype='d')
        self.points += found.size
        return self.inner(found, z)


class TestPlacingOnAGrid:
    def _placed(self, **named):
        named.setdefault('height_fn', _flat)
        named.setdefault('extent', REGION)
        named.setdefault('seed', 3)
        return scatter_on_heightfield(**named)

    def test_a_spacing_asks_for_one_per_cell(self) -> None:
        placed = self._placed(spacing=4.0)
        assert 0.8 < len(placed.positions) / (SIDE / 4.0) ** 2 < 1.2

    def test_nothing_lands_on_top_of_anything(self) -> None:
        placed = self._placed(spacing=8.0)
        points = placed.positions[:, [0, 2]]
        gaps = np.hypot(points[:, None, 0] - points[None, :, 0],
                        points[:, None, 1] - points[None, :, 1])
        np.fill_diagonal(gaps, 1e9)
        assert float(gaps.min()) > 0.5

    def test_the_set_is_not_a_lattice(self) -> None:
        """Jittered, or a forest is an orchard."""
        placed = self._placed(spacing=6.0)
        points = placed.positions[:, [0, 2]]
        gaps = np.hypot(points[:, None, 0] - points[None, :, 0],
                        points[:, None, 1] - points[None, :, 1])
        np.fill_diagonal(gaps, 1e9)
        near = gaps.min(axis=1)
        assert float(near.std()) > 0.15 * float(near.mean())

    def test_it_covers_the_whole_region(self) -> None:
        placed = self._placed(spacing=8.0)
        assert float(placed.positions[:, 0].min()) < -SIDE / 2 + 12.0
        assert float(placed.positions[:, 0].max()) > SIDE / 2 - 12.0

    def test_the_same_seed_is_the_same_forest(self) -> None:
        assert np.allclose(self._placed(spacing=5.0).positions,
                           self._placed(spacing=5.0).positions)

    def test_a_different_seed_is_a_different_one(self) -> None:
        assert not np.allclose(self._placed(spacing=5.0, seed=1).positions,
                               self._placed(spacing=5.0, seed=2).positions)

    def test_a_density_still_works(self) -> None:
        placed = self._placed(density=0.01)
        assert 0.8 < len(placed.positions) / (SIDE * SIDE * 0.01) < 1.2

    def test_one_or_the_other_is_needed(self) -> None:
        with pytest.raises(ValueError):
            self._placed()


class TestWhatItCostsToPlace:
    def test_a_grid_asks_far_fewer_questions_than_darts(self) -> None:
        """The point of the whole thing: the same forest, a fraction of the
        height lookups."""
        darts = _Counting()
        scatter_on_heightfield(darts, REGION, density=0.5, seed=3)
        grid = _Counting()
        scatter_on_heightfield(grid, REGION, spacing=4.0, seed=3)
        assert grid.points * 4 < darts.points

    def test_the_slope_is_measured_on_the_survivors(self) -> None:
        """Four more height lookups apiece, so they are not spent on candidates
        the cheap filters have already thrown away."""
        counted = _Counting()
        scatter_on_heightfield(counted, REGION, spacing=4.0, seed=3,
                               slope_limit=80.0,
                               keep=lambda points: points[:, 0] < -SIDE / 2 + 20.0)
        cells = (SIDE / 4.0) ** 2
        # One lookup for every cell, and four for each of the tenth that stayed.
        assert counted.points < cells * 2.0

    def test_and_the_answer_is_the_same_either_way(self) -> None:
        kept = scatter_on_heightfield(
            _flat, REGION, spacing=4.0, seed=3, slope_limit=80.0,
            height_range=(-1.0, 1.0),
            keep=lambda points: points[:, 0] < 0.0)
        assert len(kept.positions)
        assert float(kept.positions[:, 0].max()) < 0.0

    def test_a_filter_that_keeps_nothing_costs_nothing_after_it(self) -> None:
        counted = _Counting()
        scatter_on_heightfield(counted, REGION, spacing=4.0, seed=3,
                               slope_limit=80.0,
                               keep=lambda points: np.zeros(len(points), bool))
        assert counted.points == (SIDE / 4.0 + 1) ** 2


class TestTheShippedForest:
    def test_it_is_still_a_forest(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        placed = ProceduralWorld().scatter()
        assert 400_000 < len(placed.positions) < 800_000

    def test_and_the_trees_are_not_in_each_other(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        placed = ProceduralWorld().scatter()
        sample = placed.positions[::400][:, [0, 2]]
        gaps = np.hypot(sample[:, None, 0] - sample[None, :, 0],
                        sample[:, None, 1] - sample[None, :, 1])
        np.fill_diagonal(gaps, 1e9)
        assert float(gaps.min()) > 0.9


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


class TestMeasuringTheSlopeCheaply:
    """Four more height lookups apiece is the most expensive filter there is,
    and the conformed height function is the most expensive thing to ask. A
    caller that already has the ground sampled -- a bake builds a height field
    for the terrain before it scatters anything on it -- can answer far more
    cheaply, and for "is this too steep for a tree" it answers better: a
    central difference over a metre reads every wrinkle of an earthwork as a
    cliff."""

    def _bank(self, x, z):
        """Flat, with a wall down the middle."""
        return np.where(np.abs(np.asarray(x, 'd')) < 20.0, 0.0, 40.0)

    def test_a_slope_function_is_used_instead(self) -> None:
        counted = _Counting(self._bank)
        scatter_on_heightfield(counted, REGION, spacing=8.0, seed=3,
                               slope_limit=30.0,
                               slope_fn=lambda x, z: np.zeros(np.shape(x)))
        assert counted.points == (SIDE / 8.0 + 1) ** 2

    def test_and_it_decides_what_stays(self) -> None:
        steep = scatter_on_heightfield(
            _flat, REGION, spacing=8.0, seed=3, slope_limit=30.0,
            slope_fn=lambda x, z: np.where(np.asarray(x, 'd') > 0,
                                           math.radians(60.0), 0.0))
        assert len(steep.positions)
        assert float(steep.positions[:, 0].max()) < 0.0

    def test_without_one_the_height_function_answers(self) -> None:
        found = scatter_on_heightfield(self._bank, REGION, spacing=8.0, seed=3,
                                       slope_limit=30.0)
        assert len(found.positions)
        assert float(np.abs(np.abs(found.positions[:, 0]) - 20.0).min()) > 1.0

    def test_the_shipped_world_uses_its_own_field(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        world = ProceduralWorld(extent=1024.0, field_resolution=257,
                                control_size=256)
        assert world.slope_fn() is not None

    def test_a_world_meshed_into_tiles_has_no_field_to_ask(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        assert ProceduralWorld(ground='tiles', extent=512.0).slope_fn() is None


class TestAskingForNone:
    """A density of nothing is how a world says it wants no trees, and a
    spacing derived from it is an infinite one -- which a grid answers with the
    single cell the whole world falls in."""

    def test_a_world_with_no_trees_has_none(self) -> None:
        from OpenGLContext_editor.world.procedural import ProceduralWorld
        bare = ProceduralWorld(tree_density=0.0, extent=512.0, ground='tiles',
                               forest='tiles', road=False)
        assert len(bare.scatter().positions) == 0

    def test_and_a_spacing_of_nothing_is_reported(self) -> None:
        with pytest.raises(ValueError):
            scatter_on_heightfield(_flat, REGION, spacing=0.0, seed=1)
