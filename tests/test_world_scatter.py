"""Placing instances on a height field, before there are any tiles."""

import math

import numpy as np
import pytest

from OpenGLContext_editor.bake.bounds import BoundingBox
from OpenGLContext_editor.world.scatter import (
    scatter_on_heightfield,
    settle_onto,
    surface_slope,
    yaw_quaternions,
)

REGION = BoundingBox((-50, 0, -50), (50, 0, 50))     # 100 x 100 m


def _flat(x, z):
    return np.zeros_like(np.asarray(x, 'd'))


def _ramp(x, z):
    """Ground rising one metre per metre eastward: a 45 degree slope."""
    return np.asarray(x, 'd')


class TestScattering:
    def test_density_sets_the_count(self) -> None:
        scatter = scatter_on_heightfield(_flat, REGION, density=0.01, seed=1)
        assert len(scatter) == 100          # 10000 m2 at 0.01/m2

    def test_placements_land_inside_the_region(self) -> None:
        scatter = scatter_on_heightfield(_flat, REGION, density=0.05, seed=1)
        assert all(REGION.contains((p[0], 0, p[2])) for p in scatter.positions)

    def test_placements_sit_on_the_ground(self) -> None:
        scatter = scatter_on_heightfield(_ramp, REGION, density=0.01, seed=2)
        assert np.allclose(scatter.positions[:, 1], scatter.positions[:, 0], atol=1e-4)

    def test_the_same_seed_places_the_same_world(self) -> None:
        first = scatter_on_heightfield(_flat, REGION, density=0.02, seed=7)
        second = scatter_on_heightfield(_flat, REGION, density=0.02, seed=7)
        assert np.array_equal(first.positions, second.positions)

    def test_a_different_seed_places_a_different_world(self) -> None:
        first = scatter_on_heightfield(_flat, REGION, density=0.02, seed=7)
        second = scatter_on_heightfield(_flat, REGION, density=0.02, seed=8)
        assert not np.array_equal(first.positions, second.positions)

    def test_yaws_and_scales_come_with_the_positions(self) -> None:
        scatter = scatter_on_heightfield(_flat, REGION, density=0.01, seed=3,
                                         scale_range=(0.8, 1.4))
        assert len(scatter.yaws) == len(scatter.positions)
        assert scatter.scales.min() >= 0.8 and scatter.scales.max() <= 1.4

    def test_zero_density_places_nothing(self) -> None:
        assert len(scatter_on_heightfield(_flat, REGION, density=0.0, seed=1)) == 0

    def test_steep_ground_can_be_excluded(self) -> None:
        assert len(scatter_on_heightfield(_ramp, REGION, density=0.05, seed=1,
                                          slope_limit=30.0)) == 0
        assert len(scatter_on_heightfield(_ramp, REGION, density=0.05, seed=1,
                                          slope_limit=50.0)) > 0

    def test_an_elevation_band_can_be_required(self) -> None:
        scatter = scatter_on_heightfield(_ramp, REGION, density=0.1, seed=1,
                                         height_range=(0.0, 10.0))
        assert len(scatter)
        assert scatter.positions[:, 1].min() >= 0.0
        assert scatter.positions[:, 1].max() <= 10.0

    def test_a_mask_can_veto_placements(self) -> None:
        """The shape a distance-to-road field takes when it thins vegetation."""
        scatter = scatter_on_heightfield(
            _flat, REGION, density=0.05, seed=1,
            keep=lambda points: points[:, 2] > 0)
        assert len(scatter)
        assert scatter.positions[:, 2].min() > 0


class TestSlope:
    def test_flat_ground_has_no_slope(self) -> None:
        assert surface_slope(_flat, np.array([0.0]), np.array([0.0]))[0] == 0.0

    def test_a_one_in_one_ramp_is_forty_five_degrees(self) -> None:
        angle = surface_slope(_ramp, np.array([0.0]), np.array([0.0]))[0]
        assert angle == pytest.approx(math.pi / 4)


class TestPlacementHelpers:
    def test_a_yaw_becomes_a_quaternion_about_up(self) -> None:
        quaternions = yaw_quaternions([0.0, math.pi])
        assert np.allclose(quaternions[0], (0, 0, 0, 1), atol=1e-6)
        assert np.allclose(quaternions[1], (0, 1, 0, 0), atol=1e-6)

    def test_points_settle_onto_the_ground(self) -> None:
        points = settle_onto(_ramp, [(3.0, 99.0, 0.0), (-2.0, -99.0, 0.0)])
        assert np.allclose(points[:, 1], (3.0, -2.0))

    def test_an_offset_lifts_them(self) -> None:
        points = settle_onto(_flat, [(0.0, 0.0, 0.0)], offset=1.5)
        assert points[0][1] == pytest.approx(1.5)
