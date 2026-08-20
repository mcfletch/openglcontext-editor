"""Real elevation on the ground: reading a height file and placing it in metres.

The facts about the file's layout and about the geodesy are in
[specs/ELEVATION-DATA.md](../specs/ELEVATION-DATA.md); what is asserted here is
that the code follows them, against a file this test writes itself and against
distances anyone can check with a calculator.
"""
import math
import struct

import numpy as np
import pytest

from OpenGLContext_editor.world.dem import (
    VOID,
    DEMBase,
    corner_from_name,
    local_frame,
    read_hgt,
)

#: A tiny stand-in for a real tile: 5 x 5 samples over one degree square.
SIDE = 5


def _write_hgt(path, samples, name='N47E008.hgt'):
    """A height file as the spec describes: big-endian int16, north row first."""
    target = path / name
    values = np.asarray(samples, dtype='>i2')
    target.write_bytes(values.tobytes())
    return str(target)


def _ramp():
    """Height rising to the north, so a reader that flips the rows is caught."""
    rows = np.arange(SIDE)[:, None] * np.ones((1, SIDE))
    return (SIDE - 1 - rows) * 100.0        # north row is the highest


class TestNamingTheSquare:
    def test_a_northern_eastern_square(self) -> None:
        assert corner_from_name('N47E008.hgt') == (47.0, 8.0)

    def test_a_southern_western_square(self) -> None:
        assert corner_from_name('S13W072.hgt') == (-13.0, -72.0)

    def test_the_directory_it_is_in_does_not_matter(self) -> None:
        assert corner_from_name('/data/srtm/N00E010.hgt') == (0.0, 10.0)

    def test_a_name_that_says_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError):
            corner_from_name('heights.hgt')


class TestReadingTheFile:
    def test_it_works_the_grid_size_out_from_the_file(self, tmp_path) -> None:
        grid = read_hgt(_write_hgt(tmp_path, _ramp()))
        assert grid.samples.shape == (SIDE, SIDE)

    def test_the_square_is_where_its_name_says(self, tmp_path) -> None:
        grid = read_hgt(_write_hgt(tmp_path, _ramp()))
        assert (grid.south, grid.west) == (47.0, 8.0)
        assert (grid.north, grid.east) == (48.0, 9.0)

    def test_the_first_row_in_the_file_is_the_north_edge(self, tmp_path) -> None:
        """A reader that stores it the other way up puts the hills in the
        valleys, and nothing about the elevations themselves says so."""
        grid = read_hgt(_write_hgt(tmp_path, _ramp()))
        assert grid.at(47.99, 8.5) > grid.at(47.01, 8.5)

    def test_a_sample_is_read_where_it_was_written(self, tmp_path) -> None:
        samples = np.zeros((SIDE, SIDE))
        samples[0, 0] = 1234.0                   # north-west corner
        grid = read_hgt(_write_hgt(tmp_path, samples))
        assert grid.at(48.0, 8.0) == pytest.approx(1234.0)

    def test_between_samples_it_interpolates(self, tmp_path) -> None:
        grid = read_hgt(_write_hgt(tmp_path, _ramp()))
        low = grid.at(47.0, 8.5)
        high = grid.at(47.25, 8.5)
        assert low < grid.at(47.125, 8.5) < high

    def test_a_void_reads_as_the_ground_around_it(self, tmp_path) -> None:
        """Rather than as thirty-two kilometres below sea level."""
        samples = np.full((SIDE, SIDE), 500.0)
        samples[2, 2] = VOID
        grid = read_hgt(_write_hgt(tmp_path, samples))
        assert grid.at(47.5, 8.5) == pytest.approx(500.0)

    def test_a_file_that_is_not_square_is_refused(self, tmp_path) -> None:
        target = tmp_path / 'N47E008.hgt'
        target.write_bytes(struct.pack('>7h', *range(7)))
        with pytest.raises(ValueError):
            read_hgt(str(target))

    def test_asking_outside_the_square_answers_at_its_edge(self, tmp_path) -> None:
        grid = read_hgt(_write_hgt(tmp_path, _ramp()))
        assert grid.at(60.0, 8.5) == pytest.approx(grid.at(48.0, 8.5))


class TestPuttingItOnTheGround:
    def test_a_degree_of_latitude_is_about_a_hundred_and_eleven_kilometres(self) -> None:
        frame = local_frame(47.0, 8.0)
        east, north = frame.metres_from(48.0, 8.0)
        assert north == pytest.approx(111_000.0, rel=0.01)
        assert east == pytest.approx(0.0, abs=1.0)

    def test_a_degree_of_longitude_shortens_towards_the_pole(self) -> None:
        at_equator = local_frame(0.0, 0.0).metres_from(0.0, 1.0)[0]
        at_sixty = local_frame(60.0, 0.0).metres_from(60.0, 1.0)[0]
        assert at_sixty == pytest.approx(at_equator * math.cos(math.radians(60.0)),
                                         rel=0.01)

    def test_east_is_positive_and_north_is_positive(self) -> None:
        east, north = local_frame(47.0, 8.0).metres_from(47.1, 8.1)
        assert east > 0 and north > 0

    def test_it_comes_back_where_it_started(self) -> None:
        frame = local_frame(47.0, 8.0)
        assert frame.degrees_from(*frame.metres_from(47.2, 8.3)) \
            == pytest.approx((47.2, 8.3), abs=1e-6)


class TestTheHeightBase:
    def _base(self, tmp_path, **named):
        path = _write_hgt(tmp_path, _ramp())
        named.setdefault('centre', (47.5, 8.5))
        return DEMBase(path=path, **named)

    def test_the_centre_of_the_world_is_the_centre_it_was_given(self, tmp_path) -> None:
        base = self._base(tmp_path)
        here = float(base.sample(np.asarray([0.0]), np.asarray([0.0]))[0])
        assert here == pytest.approx(read_hgt(base.path).at(47.5, 8.5))

    def test_north_in_the_world_is_north_on_the_ground(self, tmp_path) -> None:
        """The world's north is -z, and this landscape rises northwards."""
        base = self._base(tmp_path)
        x = np.asarray([0.0, 0.0])
        z = np.asarray([-5000.0, 5000.0])
        north, south = base.sample(x, z)
        assert north > south

    def test_the_datum_puts_the_centre_where_it_was_asked_for(self, tmp_path) -> None:
        """A world's zero is its waterline; a valley 400 m up is not 400 m of
        ground the road has to climb."""
        base = self._base(tmp_path, datum=0.0)
        assert float(base.sample(np.asarray([0.0]), np.asarray([0.0]))[0]) \
            == pytest.approx(0.0, abs=1e-6)

    def test_relief_scales_what_is_left(self, tmp_path) -> None:
        full = self._base(tmp_path, datum=0.0)
        half = self._base(tmp_path, datum=0.0, relief=0.5)
        x, z = np.asarray([0.0, 500.0]), np.asarray([-3000.0, 1000.0])
        assert np.allclose(half.sample(x, z), full.sample(x, z) * 0.5)

    def test_the_answer_is_the_shape_of_the_question(self, tmp_path) -> None:
        base = self._base(tmp_path)
        axis = np.linspace(-1000.0, 1000.0, 6)
        x, z = np.meshgrid(axis, axis, indexing='ij')
        assert base.sample(x, z).shape == x.shape

    def test_it_reads_the_file_once(self, tmp_path) -> None:
        base = self._base(tmp_path)
        base.sample(np.zeros(1), np.zeros(1))
        assert base.grid() is base.grid()

    def test_it_round_trips_through_a_project_file(self, tmp_path) -> None:
        from OpenGLContext_editor.world.height import base_from_json
        base = self._base(tmp_path, relief=0.8, datum=12.0)
        assert base_from_json(base.to_json()) == base

    def test_the_file_it_needs_is_named_in_the_project(self, tmp_path) -> None:
        base = self._base(tmp_path)
        assert base.to_json()['path'] == base.path
        assert tuple(base.to_json()['centre']) == (47.5, 8.5)
