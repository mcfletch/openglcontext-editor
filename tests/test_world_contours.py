"""Iso-height lines over a sampled height field.

Pure arithmetic on a grid: a known field has contours anybody can work out by
hand, so that is what is asserted.
"""
import numpy as np
import pytest

from OpenGLContext_editor.world.contours import (
    Contour,
    contour_levels,
    contours,
    contours_of,
)

HALF = 100.0


def _grid(fn, steps=81, half=HALF):
    axis = np.linspace(-half, half, steps)
    x, z = np.meshgrid(axis, axis, indexing='ij')
    return fn(x, z), (-half, half, -half, half)


def _cone(x, z):
    """A hill: highest in the middle, falling away in every direction."""
    return 100.0 - np.hypot(x, z)


def _ramp(x, z):
    """A plane tilted along x: contours are straight lines across it."""
    return x


class TestWhichLevelsAreDrawn:
    def test_they_are_multiples_of_the_interval(self) -> None:
        levels = contour_levels(-38.0, 91.0, interval=25.0)
        assert list(levels) == [-25.0, 0.0, 25.0, 50.0, 75.0]

    def test_a_field_with_no_range_has_none(self) -> None:
        assert list(contour_levels(12.0, 12.0, interval=10.0)) == []

    def test_an_interval_of_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError):
            contour_levels(0.0, 10.0, interval=0.0)

    def test_they_can_be_offset_from_zero(self) -> None:
        """Sea level is not always the datum a designer wants them counted from."""
        levels = contour_levels(0.0, 30.0, interval=10.0, base=2.5)
        assert list(levels) == [2.5, 12.5, 22.5]


class TestAHillsContours:
    def _contours(self, interval=20.0):
        heights, extent = _grid(_cone)
        return contours(heights, *extent, interval=interval)

    def _inside(self, interval=20.0):
        """The levels whose ring lies wholly on the ground.

        The cone is sampled over a square, so a ring wider than the square is
        four arcs running off its sides -- which is what it should be, and is
        asserted below rather than here.
        """
        return [contour for contour in self._contours(interval)
                if contour.elevation > 0.0]

    def test_each_level_is_one_closed_loop(self) -> None:
        for contour in self._inside():
            assert len(contour.lines) == 1, contour.elevation
            assert contour.closed(contour.lines[0])

    def test_every_point_on_a_line_is_at_its_elevation(self) -> None:
        for contour in self._inside():
            radius = np.hypot(contour.lines[0][:, 0], contour.lines[0][:, 1])
            assert np.allclose(100.0 - radius, contour.elevation, atol=1.0)

    def test_a_higher_contour_encloses_a_smaller_area(self) -> None:
        found = sorted(self._inside(), key=lambda c: c.elevation)
        radii = [np.hypot(c.lines[0][:, 0], c.lines[0][:, 1]).mean()
                 for c in found]
        assert radii == sorted(radii, reverse=True)

    def test_a_ring_wider_than_the_ground_comes_back_as_open_arcs(self) -> None:
        """One per corner it still crosses, and none of them closed."""
        low = [c for c in self._contours() if c.elevation == -40.0]
        assert low and len(low[0].lines) == 4
        assert not any(low[0].closed(line) for line in low[0].lines)

    def test_a_finer_interval_draws_more_of_them(self) -> None:
        assert len(self._contours(10.0)) > len(self._contours(40.0))

    def test_none_are_drawn_outside_the_ground(self) -> None:
        heights, _extent = _grid(_cone)
        for contour in self._contours():
            assert heights.min() <= contour.elevation <= heights.max()


class TestASlopesContours:
    def _contours(self, interval=25.0):
        heights, extent = _grid(_ramp)
        return contours(heights, *extent, interval=interval)

    def test_each_is_a_straight_line_at_its_own_x(self) -> None:
        for contour in self._contours():
            for line in contour.lines:
                assert np.allclose(line[:, 0], contour.elevation, atol=1e-6)

    def test_a_line_that_runs_off_the_ground_is_not_closed(self) -> None:
        for contour in self._contours():
            assert not any(contour.closed(line) for line in contour.lines)

    def test_it_spans_the_whole_ground(self) -> None:
        for contour in self._contours():
            line = np.vstack(contour.lines)
            assert line[:, 1].min() == pytest.approx(-HALF, abs=3.0)
            assert line[:, 1].max() == pytest.approx(HALF, abs=3.0)


class TestFlatGround:
    def test_nothing_is_drawn_on_a_plain(self) -> None:
        heights, extent = _grid(lambda x, z: np.zeros_like(x))
        assert contours(heights, *extent, interval=10.0) == []


class TestFromAHeightFunction:
    def test_it_samples_the_function_over_the_extent(self) -> None:
        found = contours_of(_cone, extent=2 * HALF, interval=20.0,
                            resolution=81)
        assert found and all(isinstance(c, Contour) for c in found)
        assert {c.elevation for c in found} \
            == {c.elevation for c in contours(*_grid(_cone)[:1],
                                              -HALF, HALF, -HALF, HALF,
                                              interval=20.0)}

    def test_a_coarser_sampling_still_finds_the_same_levels(self) -> None:
        fine = contours_of(_cone, extent=2 * HALF, interval=20.0, resolution=129)
        coarse = contours_of(_cone, extent=2 * HALF, interval=20.0, resolution=33)
        assert {c.elevation for c in fine} == {c.elevation for c in coarse}
