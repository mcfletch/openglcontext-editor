"""Raising and lowering ground: what one stroke of a brush does to the land.

Pure arithmetic over ``(x, z)``, so all of it is asserted directly.
"""
import numpy as np
import pytest
from OpenGLContext.loaders.tiles3d.procedural import terrain_height

from OpenGLContext_editor.world.height import HeightSource, ProceduralBase, edit_from_json
from OpenGLContext_editor.world.sculpt import SculptStroke


def _grid(half=300.0, steps=61):
    axis = np.linspace(-half, half, steps)
    return np.meshgrid(axis, axis, indexing='ij')


def _source(*edits):
    return HeightSource(base=ProceduralBase(relief=1.0), edits=list(edits))


def _difference(*edits, half=300.0, steps=61):
    x, z = _grid(half, steps)
    return _source(*edits).height_fn()(x, z) - terrain_height(x, z), x, z


class TestRaisingGround:
    def test_the_centre_goes_up_by_the_amount_asked_for(self) -> None:
        stroke = SculptStroke(centre=(0.0, 0.0), radius=100.0, amount=30.0,
                              detail=0.0)
        difference, x, z = _difference(stroke)
        middle = (np.abs(x) < 6.0) & (np.abs(z) < 6.0)
        assert difference[middle].max() == pytest.approx(30.0, abs=1.5)

    def test_a_negative_amount_lowers_it(self) -> None:
        stroke = SculptStroke(centre=(0.0, 0.0), radius=100.0, amount=-30.0,
                              detail=0.0)
        difference, x, z = _difference(stroke)
        assert difference.min() < -25.0
        assert difference.max() <= 1e-9

    def test_it_fades_to_nothing_at_the_edge_of_the_brush(self) -> None:
        stroke = SculptStroke(centre=(0.0, 0.0), radius=100.0, amount=30.0,
                              detail=0.0)
        difference, x, z = _difference(stroke)
        rim = np.abs(np.hypot(x, z) - 100.0) < 4.0
        assert np.allclose(difference[rim], 0.0, atol=1.0)

    def test_nothing_outside_the_brush_moves_at_all(self) -> None:
        stroke = SculptStroke(centre=(0.0, 0.0), radius=100.0, amount=30.0)
        difference, x, z = _difference(stroke)
        assert np.all(difference[np.hypot(x, z) > 101.0] == 0.0)

    def test_a_softer_falloff_spreads_the_same_lift_wider(self) -> None:
        sharp = SculptStroke(radius=100.0, amount=30.0, falloff=4.0, detail=0.0)
        soft = SculptStroke(radius=100.0, amount=30.0, falloff=1.0, detail=0.0)
        raised = 5.0
        assert ((_difference(soft)[0] > raised).sum()
                > (_difference(sharp)[0] > raised).sum())


class TestKeepingItNatural:
    def test_detail_breaks_up_a_smooth_dome(self) -> None:
        """A hill with no small-scale variation reads as a bubble on the map."""
        smooth = _difference(SculptStroke(radius=150.0, amount=40.0,
                                          detail=0.0))[0]
        natural = _difference(SculptStroke(radius=150.0, amount=40.0,
                                           detail=0.5))[0]
        assert natural.std() > smooth.std()

    def test_the_detail_stays_inside_the_brush(self) -> None:
        difference, x, z = _difference(
            SculptStroke(radius=100.0, amount=40.0, detail=0.8))
        assert np.all(difference[np.hypot(x, z) > 101.0] == 0.0)

    def test_it_is_a_share_of_the_lift_rather_than_a_free_amount(self) -> None:
        """So a gentle stroke gets gentle detail and does not turn to gravel."""
        big = _difference(SculptStroke(radius=150.0, amount=80.0, detail=0.5))[0]
        small = _difference(SculptStroke(radius=150.0, amount=8.0, detail=0.5))[0]
        assert big.std() > small.std() * 4.0

    def test_the_same_stroke_makes_the_same_ground_every_time(self) -> None:
        stroke = SculptStroke(radius=150.0, amount=40.0, detail=0.6, seed=3)
        assert np.array_equal(_difference(stroke)[0], _difference(stroke)[0])

    def test_a_different_seed_is_different_detail(self) -> None:
        one = _difference(SculptStroke(radius=150.0, amount=40.0, detail=0.6,
                                       seed=1))[0]
        other = _difference(SculptStroke(radius=150.0, amount=40.0, detail=0.6,
                                         seed=2))[0]
        assert not np.allclose(one, other)


class TestWhatItCosts:
    def test_it_declares_the_ground_it_can_reach(self) -> None:
        stroke = SculptStroke(centre=(100.0, -50.0), radius=40.0, amount=5.0)
        assert stroke.bounds() == (60.0, -90.0, 140.0, -10.0)

    def test_a_stroke_the_query_misses_is_never_asked(self) -> None:
        asked = []

        class Counting(SculptStroke):
            def delta(self, x, z, height):
                asked.append(len(np.ravel(x)))
                return super().delta(x, z, height)

        far = Counting(centre=(9000.0, 0.0), radius=10.0, amount=5.0)
        _source(far).height_fn()(*_grid(half=100.0, steps=9))
        assert asked == []


class TestStrokesTogether:
    def test_two_strokes_add_where_they_overlap(self) -> None:
        one = SculptStroke(centre=(-20.0, 0.0), radius=100.0, amount=20.0,
                           detail=0.0)
        other = SculptStroke(centre=(20.0, 0.0), radius=100.0, amount=20.0,
                             detail=0.0)
        both, x, z = _difference(one, other)
        middle = (np.abs(x) < 5.0) & (np.abs(z) < 5.0)
        assert both[middle].mean() > 25.0

    def test_lowering_over_a_raise_takes_it_back_off(self) -> None:
        up = SculptStroke(radius=100.0, amount=25.0, detail=0.0)
        down = SculptStroke(radius=100.0, amount=-25.0, detail=0.0)
        assert np.allclose(_difference(up, down)[0], 0.0, atol=1e-9)


class TestTheFile:
    def test_a_stroke_round_trips(self) -> None:
        stroke = SculptStroke(centre=(12.0, -7.0), radius=64.0, amount=-9.0,
                              falloff=1.5, detail=0.4, seed=8)
        assert edit_from_json(stroke.to_json()) == stroke

    def test_it_says_what_kind_it_is(self) -> None:
        assert SculptStroke().to_json()['kind'] == 'sculpt'
